class MetricsService:
    @staticmethod
    def calculate_cmi(labels: list[str]) -> float:
        """
        Calculates Code Mixing Index (CMI) based on token labels.
        Labels usually contain 'HIN', 'BEN', 'GUJ', 'ENG', 'OTHER'.
        """
        total_tokens = len(labels)
        if total_tokens == 0:
            return 0.0
            
        other_count = labels.count("OTHER")
        language_tokens = total_tokens - other_count
        
        if language_tokens == 0:
            return 0.0
            
        # Count frequencies of all language-specific tags
        lang_counts = {}
        for tag in ["HIN", "BEN", "GUJ", "ENG"]:
            lang_counts[tag] = labels.count(tag)
            
        # Max(all langs) is the matrix language
        max_lang = max(lang_counts.values())
        
        # CMI Formula: 100 * (1 - (max_lang / language_tokens))
        cmi = 100 * (1 - (max_lang / language_tokens))
        
        return round(cmi, 2)
