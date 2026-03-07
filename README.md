# Document Features Extraction

This document outlines the 10 linguistic and statistical features that will be extracted from the Reuters-21578 dataset for classification purposes.

### 1. Most Frequent Word (After Filtering)
The word that occurs most often in the text after removing conjunctions, prepositions, etc.
*   **Calculation:** Count occurrences of each remaining word and select the one with the highest count.

### 2. Longest Word
The word with the maximum number of characters in the document.
*   **Calculation:** Identify all individual words and find the one with the `max(length)`.

### 3. Vocabulary Richness 
A measure of how varied the vocabulary is in the text.
*   **Calculation:** `Number of Unique Words / Total Number of Words`

### 4. Average Word Length
The mean number of characters per word.
*   **Calculation:** `Total Number of Characters (in words) / Total Number of Words`

### 5. Average Sentence Length
The mean number of words per sentence.
*   **Calculation:** `Total Number of Words / Number of Sentences`

### 6. Uppercase Letter Ratio
The proportion of capital letters relative to the total length of the text.
*   **Calculation:** `Number of Uppercase Letters / Total Number of Characters`

### 7. Financial/Numerical Sign Density
The frequency of special characters related to finance and numbers (e.g., $, %, €).
*   **Calculation:** `Number of Financial & Numerical Characters / Total Number of Characters`

### 8. Flesch Reading Ease Index
A score that indicates how difficult a passage in English is to understand.
*   **Calculation:** `206.835 - 1.015 * (Total Words / Total Sentences) - 84.6 * (Total Syllables / Total Words)`
    *   *Note: Higher scores indicate material that is easier to read.*

### 9. Vowel to Consonant Ratio
The relationship between the number of vowels and the number of consonants in the text.
*   **Calculation:** `Number of Vowels / Number of Consonants`

### 10. Sum of All Numerical Values
The mathematical sum of all numbers found within the document.
*   **Calculation:** Extract all numerical strings (e.g., "100", "5.5"), convert them to their numeric values, and add them together.

