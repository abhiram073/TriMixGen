class MetricsService:
    @staticmethod
    def calculate_cmi(labels: list[str]) -> float:
        """
        Calculates Code Mixing Index (CMI) based on token labels.
        Labels usually contain 'EN', 'TE', 'OTHER'.
        """
        total_tokens = len(labels)
        if total_tokens == 0:
            return 0.0
            
        te_count = labels.count("TE")
        en_count = labels.count("EN")
        other_count = labels.count("OTHER")
        
        language_tokens = te_count + en_count
        if language_tokens == 0:
            return 0.0
            
        # Max(TE, EN) is the matrix language
        max_lang = max(te_count, en_count)
        
        # CMI Formula: 100 * (1 - (max_lang / language_tokens))
        cmi = 100 * (1 - (max_lang / language_tokens))
        
        return round(cmi, 2)
