#  Harry Potter NLP: Mapping Narrative Agency through Personality
**A Data-Driven Analysis of Cinematic Character Archetypes**

##  Project Overview
This project uses Natural Language Processing (NLP) to analyze the dialogue of 32 characters from the Harry Potter films. By mapping dialogue to the **Big Five Personality Traits**, I used unsupervised learning to identify narrative archetypes that move beyond the traditional "Hero vs. Villain" labels.

##  Technical Pipeline
1. **Feature Engineering**: Standardized Big Five trait scores for characters with >1,000 tokens of dialogue.
2. **Dimensionality Reduction**: Implemented **PCA** to identify the 3 primary axes of character behavior, capturing ~97% of variance.
3. **Clustering**: Applied **Agglomerative Hierarchical Clustering** using **Cosine Distance** to group characters based on the *profile* of their personality rather than just intensity.
4. **Soft-SLOAN System**: Developed a custom classification system where uppercase letters denote extreme traits (|Z| > 0.5) and lowercase letters denote moderate tendencies.

##  Key Findings
- **The "Hero" Paradox**: The main trio (Harry, Ron, Hermione) share a reactive profile (**RLUEN**), acting as emotional proxies for the audience.
- **Narrative Agency**: Cluster analysis reveals that characters like Voldemort and Dumbledore share high "Narrative Agency" (low neuroticism, high organization), regardless of their morality.
- **Linguistic Stability**: Dialogue-based analysis shows that on-screen power is mathematically correlated with linguistic stability and lack of reactivity.