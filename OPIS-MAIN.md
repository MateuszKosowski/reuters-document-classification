# Szczegółowy opis działania klasy Main

## Spis treści

1. [Ogólny przegląd](#ogólny-przegląd)
2. [Stałe konfiguracyjne](#stałe-konfiguracyjne)
3. [Przepływ główny (metoda main)](#przepływ-główny)
4. [Kluczowe techniki i struktury danych](#kluczowe-techniki-i-struktury-danych)
5. [Etapy eksperymentu](#etapy-eksperymentu)
6. [Współbieżność i przetwarzanie równoległe](#współbieżność-i-przetwarzanie-równoległe)
7. [Pozostałe metody pomocnicze](#pozostałe-metody-pomocnicze)

---

## 1. Ogólny przegląd

Klasa `Main` jest punktem wejścia aplikacji do klasyfikacji dokumentów Reuters za pomocą algorytmu k-NN (k Nearest Neighbors). Realizuje **3-etapowy pipeline eksperymentalny**, który automatycznie szuka najlepszej konfiguracji klasyfikatora:

- **Etap 1 (Stage 1):** Szukanie najlepszego `k` (liczby sąsiadów) i metryki odległości (Euclidean / Manhattan / Chebyshev)
- **Etap 2 (Stage 2):** Szukanie najlepszego podziału train/test (od 10/90 do 90/10)
- **Etap 3 (Stage 3):** Szukanie najlepszego podzbioru cech (feature selection)

Każdy etap opiera się na wynikach poprzedniego — jest to strategia **greedy sequential search** (zachłanne przeszukiwanie sekwencyjne).

---

## 2. Stałe konfiguracyjne

```java
private static final String DATA_SOURCE = "src/main/resources/reuters21578";
private static final long SEED = 42L;
private static final int MAX_K_TO_TEST = 30;
private static final double STAGE_ONE_TRAIN_RATIO = 0.50;
private static final List<Double> SPLIT_RATIOS = List.of(0.10, 0.20, ..., 0.90);
```

### Dlaczego `SEED = 42L`?

Ziarno (`seed`) gwarantuje **powtarzalność eksperymentów**. `Collections.shuffle(rawVectors, new Random(SEED))` tasuje dane w zawsze ten sam sposób. Dzięki temu każde uruchomienie daje identyczne wyniki — jest to kluczowe dla porównywalności eksperymentów naukowych.

### Dlaczego `List.of(...)` zamiast `Arrays.asList(...)`?

`List.of(...)` (Java 9+) tworzy **niemutowalną listę**:
- Próba `add()` / `remove()` wyrzuci `UnsupportedOperationException`
- Jest bezpieczniejsza wielowątkowo (immutable = thread-safe)
- Zużywa mniej pamięci (wewnętrzna implementacja jest bardziej kompaktowa)
- `Arrays.asList()` tworzy listę o stałym rozmiarze, ale pozwala na `set()` — nie jest w pełni niemutowalna

---

## 3. Przepływ główny (metoda main)

```
main()
  ├── loadAndPrepareDataset()          // Wczytanie artykułów, ekstrakcja cech, shuffle
  ├── createMetricDefinitions()         // Definicja 3 metryk odległości
  ├── precomputeNormalizedSplits()      // Prekalkulacja znormalizowanych splitów
  ├── Stage 1: searchKAndMetric()       // Grid search po k × metryka
  ├── Stage 2: searchSplits()           // Grid search po proporcjach train/test
  ├── Stage 3: searchFeatureSubsets()   // Przeszukanie podzbiorów cech
  ├── Porównanie i eksport wyników CSV
  ├── printDetailedSummary()            // Wydruk najlepszej konfiguracji
  └── shutdownExecutor()                // Zamknięcie puli wątków
```

### Blok try-finally

```java
ExecutorService executor = null;
try {
    // ... cały eksperyment ...
} finally {
    if (executor != null) {
        shutdownExecutor(executor);
    }
    logTotalExperimentTime(totalStartMs);
}
```

**Dlaczego `try-finally` zamiast `try-with-resources`?**

`ExecutorService` nie implementuje `AutoCloseable` w Java 17 (dopiero od Java 19 ma metodę `close()`). Dlatego stosujemy ręczny `finally` blok, który **zawsze się wykona** — nawet gdy zostanie rzucony wyjątek. Gwarantuje to:
- Zamknięcie puli wątków (`shutdownExecutor`)
- Wypisanie całkowitego czasu eksperymentu (`logTotalExperimentTime`)

---

## 4. Kluczowe techniki i struktury danych

### 4.1 EnumSet — wydajna kolekcja dla enumów

```java
EnumSet<FeatureName> fullFeatureSet = EnumSet.allOf(FeatureName.class);
```

#### Co to jest `EnumSet`?

`EnumSet` to specjalizowana implementacja `Set` przeznaczona **wyłącznie** dla typów wyliczeniowych (`enum`). Wewnętrznie używa **wektora bitowego** (bit vector), czyli jednej liczby `long` (64 bity), gdzie każdy bit odpowiada jednemu elementowi enuma.

#### Jak to działa wewnętrznie?

Dla enuma `FeatureName` z 10 wartościami:

```
Bit:  9  8  7  6  5  4  3  2  1  0
      │  │  │  │  │  │  │  │  │  │
      │  │  │  │  │  │  │  │  │  └─ LONGEST_WORD
      │  │  │  │  │  │  │  │  └──── MOST_FREQUENT_WORD
      │  │  │  │  │  │  │  └─────── AVERAGE_WORD_LENGTH
      │  │  │  │  │  │  └────────── VOCABULARY_RICHNESS
      │  │  │  │  │  └───────────── AVERAGE_SENTENCE_LENGTH
      │  │  │  │  └──────────────── UPPERCASE_LETTER_RATIO
      │  │  │  └─────────────────── FINANCIAL_SIGN_DENSITY
      │  │  └────────────────────── FLESCH_READING_EASE_INDEX
      │  └───────────────────────── VOWEL_TO_CONSONANT_RATIO
      └──────────────────────────── SUM_OF_ALL_NUMERIC_VALUES
```

- `EnumSet.allOf(FeatureName.class)` → `1111111111` (binarne) = wszystkie bity ustawione
- `EnumSet.noneOf(FeatureName.class)` → `0000000000` = żaden bit
- `EnumSet.of(LONGEST_WORD, AVERAGE_WORD_LENGTH)` → `0000000101`

#### Porównanie z alternatywami

| Operacja | `EnumSet` | `HashSet<FeatureName>` | `TreeSet<FeatureName>` |
|----------|-----------|------------------------|------------------------|
| `contains()` | O(1) — sprawdzenie bitu | O(1) amortyzowane — hash | O(log n) — drzewo |
| `add()` | O(1) — ustawienie bitu | O(1) amortyzowane | O(log n) |
| Pamięć (10 elementów) | **8 bajtów** (jeden `long`) | ~400+ bajtów (tablica + obiekty Entry) | ~500+ bajtów (węzły drzewa) |
| Iteracja | O(n) — skan bitów | O(capacity) — skan tablicy | O(n) — in-order traversal |
| Thread-safe? | Nie (jak każdy Set) | Nie | Nie |
| Zachowuje kolejność? | **Tak** (ordinal enuma) | Nie | Tak (naturalna) |

**Dlaczego `EnumSet` jest lepszy?**

1. **50-100× mniej pamięci** niż `HashSet` — bo cały zbiór to jeden `long`
2. **Operacje bitowe** (AND, OR, XOR) zamiast hashowania — ekstremalnie szybkie
3. **Zachowuje naturalną kolejność** enuma (wg `ordinal()`)
4. Kopiowanie: `EnumSet.copyOf()` to po prostu skopiowanie jednego `long` — O(1)

#### Gdzie jest to używane w kodzie?

- `EnumSet.allOf(FeatureName.class)` — pełny zbiór cech (etap 1 i 2)
- `EnumSet.noneOf(FeatureName.class)` — pusty zbiór jako punkt startowy do budowania podzbiorów
- `EnumSet.copyOf(activeFeatures)` — kopiowanie zbioru aktywnych cech do wyniku eksperymentu (defensywna kopia)

---

### 4.2 Maski bitowe (bitmask) — generowanie podzbiorów cech

```java
private static List<EnumSet<FeatureName>> generateFeatureSubsets() {
    FeatureName[] features = FeatureName.values();
    int maxMask = (1 << features.length) - 1;   // 2^10 - 1 = 1023

    for (int mask = 1; mask <= maxMask; mask++) {
        int featureCount = Integer.bitCount(mask);
        if (featureCount == features.length) continue;

        EnumSet<FeatureName> subset = EnumSet.noneOf(FeatureName.class);
        for (int bitIndex = 0; bitIndex < features.length; bitIndex++) {
            if ((mask & (1 << bitIndex)) != 0) {
                subset.add(features[bitIndex]);
            }
        }
        subsets.add(subset);
    }
}
```

#### Jak działa generowanie podzbiorów za pomocą masek bitowych?

Jest to technika **power set generation** (generowanie zbioru potęgowego). Dla `n` elementów istnieje dokładnie `2^n` podzbiorów (włącznie z pustym). Każdy podzbiór kodujemy jako liczbę od `0` do `2^n - 1`, gdzie każdy bit odpowiada obecności danego elementu.

#### Przykład dla 3 cech (A, B, C):

```
mask = 0  →  000  →  {}           (podzbiór pusty — pominięty, bo mask=0 nie wchodzi do pętli)
mask = 1  →  001  →  {A}
mask = 2  →  010  →  {B}
mask = 3  →  011  →  {A, B}
mask = 4  →  100  →  {C}
mask = 5  →  101  →  {A, C}
mask = 6  →  110  →  {B, C}
mask = 7  →  111  →  {A, B, C}   (pominięty — pełny zbiór)
```

#### Kluczowe operacje bitowe:

1. **`1 << features.length`** — przesunięcie bitowe w lewo. `1 << 10` = 1024 (`10000000000` binarnie)
2. **`(1 << features.length) - 1`** — maxMask = 1023 (`1111111111`), czyli maska z wszystkimi bitami ustawionymi
3. **`mask & (1 << bitIndex)`** — sprawdzenie, czy bit na pozycji `bitIndex` jest ustawiony. Operacja AND: jeśli wynik != 0, to bit jest "włączony" → cecha jest w podzbiorze
4. **`Integer.bitCount(mask)`** — wbudowana metoda JDK, która liczy ile bitów jest ustawionych. Na wielu procesorach mapuje się na instrukcję maszynową `POPCNT` — wykonuje się w jednym cyklu zegara!

#### Dlaczego maski bitowe zamiast rekurencji?

| Podejście | Zalety | Wady |
|-----------|--------|------|
| **Maski bitowe** | Prosta pętla for, brak alokacji stosu, łatwe do zrównoleglenenia | Limit: max 64 elementy (rozmiar `long`) |
| **Rekurencja** | Naturalna dla drzew decyzyjnych | Zużycie stosu O(n), trudniejsze w debugowaniu |
| **Iterator combinatoric** (np. Guava) | Elastyczny, lazy evaluation | Zewnętrzna zależność, overhead |

Dla 10 cech (1023 podzbiorów) maski bitowe są **idealnym wyborem** — proste, szybkie i nie wymagają dodatkowych bibliotek.

#### Sortowanie podzbiorów:

```java
subsets.sort(Comparator
    .<EnumSet<FeatureName>>comparingInt(Set::size)
    .reversed()
    .thenComparing(featureSubsetService::formatFeatureSet));
```

Sortuje od największych podzbiorów (9 cech) do najmniejszych (1 cecha), a przy równym rozmiarze — alfabetycznie po etykietach. Dzięki temu wyniki CSV są deterministyczne i czytelne.

---

### 4.3 Java Records — niemutowalne obiekty danych

```java
public record FeatureVector(
    String label,
    String longestWord,
    String mostFrequentWord,
    double averageWordLength,
    // ...
) {}
```

#### Co to jest `record` (Java 16+)?

`record` to specjalny typ klasy, który automatycznie generuje:
- **Konstruktor kanoniczny** (z wszystkimi polami)
- **Metody dostępowe** (np. `label()` zamiast `getLabel()` — bez prefixu `get`)
- **`equals()`** i **`hashCode()`** — porównanie wartościowe po wszystkich polach
- **`toString()`** — czytelna reprezentacja tekstowa

#### Dlaczego record zamiast zwykłej klasy?

```java
// BEZ record — trzeba napisać ~50 linii kodu:
public class FeatureVector {
    private final String label;
    private final double averageWordLength;
    // ... 10 pól
    
    public FeatureVector(String label, double averageWordLength, ...) {
        this.label = label;
        this.averageWordLength = averageWordLength;
        // ...
    }
    
    public String getLabel() { return label; }
    public double getAverageWordLength() { return averageWordLength; }
    // ... 10 getterów
    
    @Override public boolean equals(Object o) { /* ~15 linii */ }
    @Override public int hashCode() { /* ~5 linii */ }
    @Override public String toString() { /* ~5 linii */ }
}

// Z record — 1 linia:
public record FeatureVector(String label, double averageWordLength, ...) {}
```

#### Kluczowa cecha: niemutowalność

Pola record są zawsze `final` — po utworzeniu obiektu nie można ich zmienić. To jest kluczowe dla:
- **Bezpieczeństwa wielowątkowego** — immutable obiekty mogą być bezpiecznie współdzielone między wątkami bez synchronizacji
- **Przewidywalności** — wiemy, że wektor cech nie zmieni się po utworzeniu

W tym projekcie wszystkie modele to records: `FeatureVector`, `SingleArticle`, `ExperimentOutcome`, `DatasetSplit`, `ClassMetrics`, `CompletedExperiment`, `ExperimentTask`, `MetricDefinition`, `DetailedMetrics`, `Neighbor`, `StageSearchResult`.

---

### 4.4 TreeSet — posortowany zbiór

```java
List<Double> requiredRatios = new ArrayList<>(new TreeSet<>(SPLIT_RATIOS));
```

#### Co robi ten fragment?

1. `new TreeSet<>(SPLIT_RATIOS)` — tworzy **posortowany zbiór** z listy (automatycznie sortuje i usuwa duplikaty)
2. `new ArrayList<>(...)` — konwertuje z powrotem na listę (bo potrzebujemy `.add()` i `.sort()`)

#### Dlaczego `TreeSet` zamiast `HashSet`?

| Cecha | `TreeSet` | `HashSet` |
|-------|-----------|-----------|
| Kolejność | **Posortowana** (naturalna lub Comparator) | Brak gwarancji kolejności |
| `add/remove/contains` | O(log n) | O(1) amortyzowane |
| Implementacja | Czerwono-czarne drzewo | Tablica hashowa |
| Użycie tutaj | Deduplikacja + sortowanie w jednym kroku | Wymagałby osobnego `Collections.sort()` |

W tym kodzie `TreeSet` jest użyty jako **jednorazowy transformer**: sortuje ratios i eliminuje duplikaty w jednym kroku.

Również w `calculateDetailedMetrics()`:
```java
Set<String> classNames = new TreeSet<>();
```
Tutaj `TreeSet` gwarantuje, że nazwy klas będą **zawsze w tej samej kolejności alfabetycznej** — ważne dla deterministycznych wyników CSV i raportów.

---

### 4.5 Collectors.toCollection(ArrayList::new) — mutowalny wynik ze streama

```java
List<FeatureVector> rawVectors = rawArticles.parallelStream()
    .map(...)
    .filter(Objects::nonNull)
    .collect(Collectors.toCollection(ArrayList::new));
```

#### Dlaczego nie zwykły `.toList()`?

- `.toList()` (Java 16+) zwraca **niemutowalną** listę — nie można potem zrobić `shuffle()`
- `.collect(Collectors.toList())` — technicznie mutowalny, ale dokumentacja nie gwarantuje tego
- **`Collectors.toCollection(ArrayList::new)`** — jawnie tworzy `ArrayList`, który na pewno jest mutowalny

Jest to potrzebne, bo zaraz potem wywołujemy:
```java
Collections.shuffle(rawVectors, new Random(SEED));
```
`shuffle()` modyfikuje listę in-place i wymaga mutowalnej listy.

---

### 4.6 parallelStream — równoległa ekstrakcja cech

```java
List<FeatureVector> rawVectors = rawArticles.parallelStream()
    .map(article -> extractor.extractAllFeaturesFromArticle(article))
    .filter(Objects::nonNull)
    .collect(Collectors.toCollection(ArrayList::new));
```

#### Jak działa `parallelStream()`?

`parallelStream()` używa wewnętrznego **ForkJoinPool** JVM (Common Pool). Automatycznie dzieli kolekcję na fragmenty i przetwarza je na wielu wątkach:

```
[article1, article2, article3, article4, article5, article6]
      ↓ Fork
[art1, art2, art3]          [art4, art5, art6]
    Thread-1                    Thread-2
      ↓ map()                    ↓ map()
[vector1, vector2, vector3]  [vector4, vector5, vector6]
      ↓ Join
[vector1, vector2, vector3, vector4, vector5, vector6]
```

#### Kiedy warto używać `parallelStream()`?

- ✅ Gdy operacja na każdym elemencie jest **kosztowna** (tu: ekstrakcja cech z tekstu — parsowanie, obliczenia)
- ✅ Gdy elementów jest **dużo** (tu: tysiące artykułów)
- ❌ NIE używać dla prostych operacji (np. sumowanie listy 100 intów) — overhead tworzenia wątków > zysk

---

## 5. Etapy eksperymentu

### 5.1 Etap 1: searchKAndMetric — przeszukiwanie k i metryk

**Cel:** Dla ustalonego podziału 50/50 i wszystkich cech, znaleźć najlepsze `(k, metryka)`.

**Przestrzeń przeszukiwania:** `k ∈ {1..30}` × `metryka ∈ {Euclidean, Manhattan, Chebyshev}` = **90 kombinacji**.

Wszystkie 90 zadań są tworzone jako lista `ExperimentTask` i przesyłane do puli wątków. Każde zadanie jest niezależne, więc mogą działać równolegle.

#### Wzorzec "single-element array" jako mutowalny holder:

```java
final ExperimentOutcome[] bestOutcomeHolder = new ExperimentOutcome[1];
```

**Dlaczego tablica jednoelementowa?**

Lambdy w Javie mogą przechwytywać tylko zmienne `final` lub effectively final. Nie można napisać:
```java
ExperimentOutcome best = null;  // effectively final
consumer = outcome -> { best = outcome; };  // BŁĄD KOMPILACJI! Modyfikacja zmiennej
```

Rozwiązanie: tablica jednoelementowa jest `final` (referencja do tablicy się nie zmienia), ale jej **zawartość** może być modyfikowana:
```java
final ExperimentOutcome[] holder = new ExperimentOutcome[1];  // final
consumer = outcome -> { holder[0] = outcome; };  // OK! Modyfikujemy zawartość, nie referencję
```

**Alternatywy:**
- `AtomicReference<ExperimentOutcome>` — bezpieczniejsze wielowątkowo, ale tu callback jest wywoływany sekwencyjnie (jeden po drugim przez `completionService.take()`), więc synchronizacja nie jest konieczna
- Zwykłe pole klasy — ale Main jest klasą statyczną
- Zmienna lokalna — nie zadziała (wymóg effectively final)

### 5.2 Etap 2: searchSplits — przeszukiwanie proporcji podziału

**Cel:** Dla najlepszego `(k, metryka)` z etapu 1, znaleźć najlepszy podział train/test.

**Przestrzeń przeszukiwania:** 9 proporcji (10/90, 20/80, ..., 90/10).

Używa **prekalkuowanych splitów** (`preparedSplits` — `Map<Double, DatasetSplit>`), żeby nie normalizować danych wielokrotnie.

### 5.3 Etap 3: searchFeatureSubsets — przeszukiwanie podzbiorów cech

**Cel:** Dla najlepszego `(k, metryka, split)`, sprawdzić czy podzbiór cech daje lepszy wynik.

**Przestrzeń przeszukiwania:** `2^10 - 2 = 1022` podzbiorów (bez pustego i pełnego zbioru).

Tutaj wykorzystywane jest **maskowanie cech** — zamiast usuwać kolumny z wektora cech, wyzerowujemy wartości nieaktywnych cech:

```java
// Maskowanie: nieaktywne cechy → 0.0 (numeryczne) lub "" (tekstowe)
activeFeatures.contains(FeatureName.AVERAGE_WORD_LENGTH) ? vector.averageWordLength() : 0.0
```

**Dlaczego maskowanie zamiast usunięcia?**

Record `FeatureVector` ma stałą strukturę (zawsze 10 pól). Nie można "usunąć" pola z record. Maskowanie (zerowanie) jest semantycznie równoważne z usunięciem — zerowa wartość po normalizacji nie wpływa na odległość.

---

## 6. Współbieżność i przetwarzanie równoległe

### 6.1 ExecutorService i Fixed Thread Pool

```java
int threadPoolSize = Math.max(1, Runtime.getRuntime().availableProcessors());
executor = Executors.newFixedThreadPool(threadPoolSize);
```

#### Co to jest `ExecutorService`?

`ExecutorService` to abstrakcja Javy nad pulą wątków. Zamiast ręcznie tworzyć wątki (`new Thread()`), przekazujemy zadania do puli, a ona zarządza wątkami.

#### Dlaczego `newFixedThreadPool`?

| Typ puli | Opis | Kiedy używać |
|----------|------|-------------|
| `newFixedThreadPool(n)` | **Stała liczba n wątków** | CPU-bound tasks (obliczenia) ✅ |
| `newCachedThreadPool()` | Dynamiczna pula, rośnie i maleje | I/O-bound tasks (sieć, dysk) |
| `newSingleThreadExecutor()` | Jeden wątek | Sekwencyjne przetwarzanie |
| `newVirtualThreadPerTaskExecutor()` | Virtual threads (Java 21+) | Masowe I/O operacje |

Tu używamy `fixedThreadPool`, bo klasyfikacja k-NN to **zadanie CPU-bound** — każdy eksperyment intensywnie oblicza odległości. Liczba wątków = liczba rdzeni CPU (`availableProcessors()`).

### 6.2 ExecutorCompletionService — odbieranie wyników w kolejności zakończenia

```java
ExecutorCompletionService<CompletedExperiment> completionService = new ExecutorCompletionService<>(executor);

// Wysyłanie zadań
for (ExperimentTask task : tasks) {
    completionService.submit(() -> { ... });
}

// Odbieranie wyników W KOLEJNOŚCI ZAKOŃCZENIA
for (int completed = 1; completed <= tasks.size(); completed++) {
    Future<CompletedExperiment> future = completionService.take();  // blokuje aż coś się skończy
    CompletedExperiment result = future.get();
    onCompleted.accept(result);
}
```

#### Kluczowa różnica: `CompletionService` vs zwykłe `Future`

**Bez CompletionService:**
```java
List<Future<Result>> futures = tasks.stream()
    .map(executor::submit)
    .toList();

// Problem: czekamy na futures[0], nawet jeśli futures[5] już się skończyło!
for (Future<Result> f : futures) {
    Result r = f.get();  // blokuje w kolejności WYSŁANIA
}
```

**Z CompletionService:**
```java
completionService.take();  // zwraca PIERWSZY ZAKOŃCZONY wynik, niezależnie od kolejności wysłania
```

Dzięki temu:
- Wyniki przetwarzamy **jak najszybciej** (immediately on completion)
- Logowanie postępu jest bardziej responsywne
- Nie marnujemy czasu na czekanie na wolne zadanie, gdy szybkie już się zakończyły

#### Wzorzec Consumer callback:

```java
private static void consumeExperimentTasks(
    ExecutorService executor,
    List<ExperimentTask> tasks,
    String progressPrefix,
    int logEvery,
    Consumer<CompletedExperiment> onCompleted)
```

Metoda `consumeExperimentTasks` jest **generyczna** — przyjmuje `Consumer<CompletedExperiment>` jako callback. Każdy etap (1, 2, 3) przekazuje swoją logikę obsługi wyniku jako lambda:

```java
consumeExperimentTasks(executor, tasks, "Stage 1", 30,
    completedTask -> {
        // Logika specyficzna dla Stage 1
        appendCsvRows(csvRows, "STAGE_1_K_METRIC", outcome);
        if (isBetter(outcome, bestOutcomeHolder[0])) { ... }
    }
);
```

To jest wzorzec **Strategy** zaimplementowany przez interfejs funkcyjny `Consumer<T>`.

### 6.3 ExperimentTask i Callable

```java
public record ExperimentTask(String label, Callable<ExperimentOutcome> callable) {}
```

#### `Callable<V>` vs `Runnable`

| Cecha | `Runnable` | `Callable<V>` |
|-------|-----------|---------------|
| Zwraca wartość? | ❌ `void run()` | ✅ `V call()` |
| Rzuca checked exception? | ❌ | ✅ `throws Exception` |
| Użycie z ExecutorService | `execute()` | `submit()` → `Future<V>` |

`Callable` jest konieczne, bo eksperyment:
1. Zwraca wynik: `ExperimentOutcome`
2. Może rzucić wyjątek (np. błąd klasyfikacji)

---

## 7. Pozostałe metody pomocnicze

### 7.1 Prekalkulacja znormalizowanych splitów

```java
Map<Double, DatasetSplit> preparedSplits = precomputeNormalizedSplits(dataset, normalizationService);
```

**Cel optymalizacyjny:** Normalizacja danych (min-max scaling) jest kosztowna. Zamiast normalizować dane w każdym eksperymencie od nowa, normalizujemy raz dla każdego ratio i przechowujemy wyniki w mapie.

**Kluczowy detal:** Normalizacja zbioru testowego używa **min/max ze zbioru treningowego** (nie testowego!). To zapobiega **data leakage** — informacja ze zbioru testowego nie wpływa na przetwarzanie.

### 7.2 isBetter — porównywanie wyników

```java
private static boolean isBetter(ExperimentOutcome candidate, ExperimentOutcome currentBest) {
    if (currentBest == null) return true;
    if (candidate.metrics().selectionScore() != currentBest.metrics().selectionScore()) {
        return candidate.metrics().selectionScore() > currentBest.metrics().selectionScore();
    }
    return candidate.activeFeatures().size() < currentBest.activeFeatures().size();
}
```

Priorytet porównania:
1. Wyższy `selectionScore` (średnia harmoniczna accuracy i macro F1) wygrywa
2. Przy remisie — mniejsza liczba cech wygrywa (zasada **Occam's Razor** / prostoty modelu)

### 7.3 selectionScore — średnia harmoniczna

```java
private static double harmonicMean(double first, double second) {
    return 2.0 * first * second / (first + second);
}
```

**Dlaczego średnia harmoniczna zamiast arytmetycznej?**

Średnia harmoniczna "karze" za duże dysproporcje. Jeśli accuracy = 0.95, ale macroF1 = 0.10:
- Średnia arytmetyczna: (0.95 + 0.10) / 2 = **0.525** — wygląda "przyzwoicie"
- Średnia harmoniczna: 2 × 0.95 × 0.10 / (0.95 + 0.10) = **0.181** — ujawnia problem

To zapobiega sytuacji, gdzie model ma wysokie accuracy (bo poprawnie klasyfikuje dominującą klasę), ale słaby F1 (bo nie radzi sobie z mniejszymi klasami).

### 7.4 shutdownExecutor — prawidłowe zamykanie puli wątków

```java
private static void shutdownExecutor(ExecutorService executor) {
    executor.shutdown();                          // Krok 1: nie przyjmuj nowych zadań
    try {
        if (!executor.awaitTermination(1, TimeUnit.MINUTES)) {
            executor.shutdownNow();               // Krok 2: przerwij działające zadania
        }
    } catch (InterruptedException exception) {
        executor.shutdownNow();                   // Krok 3: przerwij przy interrupcie
        Thread.currentThread().interrupt();        // Krok 4: przywróć flagę interrupt
    }
}
```

To jest **kanoniczny wzorzec** zamykania `ExecutorService` z dokumentacji Oracle:
1. `shutdown()` — grzeczne zamknięcie (dokończ bieżące zadania)
2. `awaitTermination()` — czekaj do 1 minuty
3. `shutdownNow()` — wymuś zamknięcie jeśli nie zdążyło
4. `Thread.currentThread().interrupt()` — przywrócenie flagi przerwania, żeby wyższy kod mógł na nią zareagować

### 7.5 formatDuration — formatowanie czasu

```java
long hours = durationMs / 3_600_000;
long minutes = (durationMs % 3_600_000) / 60_000;
```

**Uwaga na literały z podkreśleniami:** `3_600_000` to to samo co `3600000` — podkreślenia (`_`) są separatorami wizualnymi (Java 7+), nie mają wpływu na wartość. Poprawiają czytelność dużych liczb:
- `3_600_000` → od razu widać "3,6 miliona" (ms w godzinie)
- `3600000` → trzeba liczyć zera

---

## 8. Struktura po refaktoryzacji

Po oczyszczeniu kodu, metody pomocnicze zostały przeniesione do dedykowanych serwisów:

| Serwis | Odpowiedzialność | Przeniesione metody z Main |
|--------|-----------------|---------------------------|
| `FeatureSubsetService` | Maskowanie cech, generowanie podzbiorów | `generateFeatureSubsets()`, `maskDataset()`, `maskVector()`, `isFullFeatureSet()`, `formatFeatureSet()`, `safeText()` |
| `ResultsExportService` | Formatowanie i eksport CSV | `csvHeader()`, `formatCsvRow()`, `appendCsvRows()` |
| `QualityMeasureService` | Obliczanie metryk jakości | `calculateDetailedMetrics()`, `harmonicMean()`, `filterClassNames()`, `isClassName()`, `extractAccuracyValue()`, `extractMacroValue()` |

**Main** zachował jedynie:
- Logikę orkiestracji pipeline'u (metoda `main`)
- Metody etapowe (`searchKAndMetric`, `searchSplits`, `searchFeatureSubsets`)
- Zarządzanie współbieżnością (`consumeExperimentTasks`, `shutdownExecutor`)
- Ładowanie danych (`loadAndPrepareDataset`)
- Logowanie i wyświetlanie wyników (`printDetailedSummary`, `logTotalExperimentTime`, `formatDuration`)

Redukcja: **859 → 683 linii** (usunięto ~175 linii z Main, przeniesiono do serwisów).
