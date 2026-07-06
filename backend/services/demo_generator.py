import random
import logging

logger = logging.getLogger(__name__)

class DemoGenerator:
    """
    Temporary fallback generator for DEMO_MODE.
    Generates realistic Telugu-English code-mixed text using pre-curated templates.
    Used when the actual MT5 model is untrained or unavailable.
    """
    
    # 10 Domains * 10 variations = 100+ realistic templates
    TEMPLATES = {
        "movie": [
            "Movie chala bagundi bro! Acting peaks asalu. Songs super unnayi. Definitely worth watching.",
            "First half koncham slow ga undi, but second half is a blockbuster. Climax was amazing.",
            "Disappointed! Story em ledu, graphics are average. Not worth the ticket price.",
            "Direction chala baagundi. Cinematography top notch. Pakka family entertainer idhi.",
            "Emaithey emundi, BGM was crazy! Theatres lo goosebumps guarantee.",
            "Hero elevation scenes vere level. Screenplay could be better though.",
            "Just average. One time watchable movie, but expectations meet avvaledu.",
            "Masterpiece! Must watch in IMAX. Visuals are stunning and mind blowing.",
            "Routine story, em kotha ga ledu. Comedy scenes only worked out well.",
            "Total paisa vasool! Entertainment overload. Weekend perfect plan idhi."
        ],
        "restaurant": [
            "Food awesome undi. Service kuda fast ga undi. Ambience is very pleasant.",
            "Biryani taste super, but spicy ga undi. Portions are totally worth it.",
            "Overpriced! Taste average, nothing special. Won't recommend going here.",
            "Starters chala baagunnayi. Main course was decent. Ambience peaks.",
            "Worst experience! Service was very slow and food cold ga vachindi.",
            "Desserts are a must try. Atmosphere chala lively ga undi. Loved it.",
            "Good place for family dinners. Staff behavior was very polite.",
            "Authentic taste! Spices perfect ga balance chesaru. Craving for more.",
            "Too much crowd. Waiting time ekkuva, but food definitely makes up for it.",
            "Quality is dropping day by day. Not the same as before. Disappointed."
        ],
        "college": [
            "Exams daggara padutunnayi. Preparation inka start cheyaledu bro. Tension ga undi.",
            "Assignments tho champestunnaru. Deadlines are too tight this semester.",
            "Class bunk kotti canteen ki veldam ra. Ee lecture chala boring.",
            "Results vachesayi! Pass ayyanu bhayya. Party pakka ivala.",
            "Placements kosam prepare avvali. Coding practice daily chestunna.",
            "Lab internals lo marks taggipoyayi. External lo cover cheyali ela aina.",
            "Fest ki arrangements ela unnayi? This time grand ga cheyali ra.",
            "Syllabus chala peddadi. One night batting tho complete avvadu idhi.",
            "Library lo chadvukundam ra. Hostel lo concentration kudaratledu.",
            "Farewell party ki dress code enti? Planning for traditional wear."
        ],
        "tech": [
            "Laptop performance chala smooth undi. Battery life kuda bagundi. Best buy.",
            "New update vachindi, kani bugs unnayi. Phone is heating up very fast.",
            "Camera quality excellent! Low light lo photos chala clear ga vastunnayi.",
            "Price ki thagga specs ivaledu. Overpriced model idhi frankly speaking.",
            "Processor speed super! Gaming experience chala lag-free ga undi.",
            "Display resolution peaks. Watching movies on this is a visual treat.",
            "Software chala clean ga undi. No bloatware, UI is very responsive.",
            "Design bagundi kani build quality feels cheap. Plastic body is a letdown.",
            "Fingerprint sensor chala fast. Face unlock kuda accurate ga pani chestundi.",
            "Fast charging support is a lifesaver. 0 to 100 in just 30 mins, crazy!"
        ],
        "social": [
            "Haha same feeling bro! Relatable AF.",
            "Ee meme chusi navvaleka sacha. Too good asalu.",
            "Congratulations mowa! Treat eppudu mari?",
            "Ignore the haters. Nuvvu nee pani chesko, you are doing great.",
            "Missing those days ra. Throwback to the best trip ever.",
            "Incredible click! Ekkada teesavu e photo?",
            "Totally agree with this point. Well said.",
            "Em cheptunnav bro? Asalu sense make avvatledu nee tweet.",
            "Waiting for the update! Eagerly looking forward to this.",
            "Happy birthday! God bless you. Party hard!"
        ],
        "travel": [
            "Goa trip plan cheddam ra. Tickets check cheyi okasari.",
            "Mountains lo weather chala chill ga undi. Best vacation ever.",
            "Resort chala peaceful ga undi. Perfect weekend getaway.",
            "Flight delay ayyindi. Airport lo waiting nundi chiraku vastundi.",
            "Views are breathtaking. Photos lo kanna real ga inka bagundi.",
            "Local food try chesam, chala spicy but tasty. Must try street food.",
            "Trekking experience was adventurous! Alisipoyanu completely.",
            "Budget trip kani enjoyment matram peaks. Covered all places.",
            "Beach sunset view is just mesmerizing. Relaxing ga undi.",
            "Packing inka avvaledu. Tomorrow early morning start avvali."
        ],
        "shopping": [
            "Dress fitting perfect ga undi. Material quality is also very nice.",
            "Sale lo theeskunna, huge discount vachindi. Totally worth it.",
            "Delivery chala late ayyindi. Customer service is not responding.",
            "Color mismatch undi. Return request raise chesa already.",
            "Sneakers look super cool. Comfort kuda bagundi try chesa.",
            "Price chala ekkuva for this quality. Don't buy this guys.",
            "Sizes available levu. Out of stock antundi app lo.",
            "Combo offer lo theeskunte best. Save money sensibly.",
            "Packing damage ayyindi kani product safe ga ne undi.",
            "Reviews chusi bayapadda kani product is genuine and good."
        ],
        "education": [
            "Online classes kanna offline e better. Concentration miss avvatledu.",
            "Ee concept chala tricky ga undi. You tube lo tutorial chudali.",
            "Notes share cheyava please? Nenu class miss ayyanu ninna.",
            "Project deadline extend chesaru. Big relief asalu.",
            "Faculty chala supportive ga unnaru. Doubts clear ga explain chestunnaru.",
            "Exams pattern change ayyindi. Preparation strategy marchali inka.",
            "Group study plan cheddam. Discuss cheste easy ga gurthuntundi.",
            "Internship apply chesava? Resume update cheyali first.",
            "Practical exams ki external evaru vastunnaro telisa?",
            "Graduation day kosam waiting! Finally finishing this degree."
        ],
        "support": [
            "Hi, na order inka dispatch kaledu. Issue enti check cheyandi.",
            "Refund status pending ani vastundi. Ee time ki credit avtundi?",
            "Account block ayyindi. Unlock cheyadaniki process enti?",
            "App crash avtundi continuously. Update vachinda emaina?",
            "Payment successful but ticket generate kaledu. Help me with this.",
            "Thanks for the quick response. Issue resolve ayyindi now.",
            "Wrong item received. Replacement process initiate cheyandi.",
            "Subscription cancel cheyali. Steps cheptara please?",
            "Login avvatledu, invalid credentials antundi. Password reset chesa kuda.",
            "Excellent support! Problem ventane fix chesaru. Thank you."
        ],
        "general": [
            "Eroju weather chala bagundi. Varsham pade la undi.",
            "Traffic chala ekkuva undi ivala. Intiki velle sariki late avtundi.",
            "Weekend em plans unnayi? Ekkadikaina bayatiki veldama?",
            "Chala rojulaki kalisam. Time asalu ela aipoindo teliyaledu.",
            "Work from home bore kodutundi. Office ke vellali inka.",
            "Emaina kotha movies vachaya OTT lo? Suggestions ivvandi.",
            "Health jagratta. Fever seasons nadustundi bayata.",
            "News chusava? Chala interesting updates vastunnayi.",
            "Eroju full busy schedule. Asalu time dorakatledu.",
            "Nidra vastundi baga. Ivala thondarga padukovali."
        ]
    }

    @staticmethod
    def infer_domain(prompt: str) -> str:
        """Heuristically infers the domain from the prompt string."""
        prompt = prompt.lower()
        if any(w in prompt for w in ["movie", "film", "cinema", "actor", "director", "theatre", "screen"]): return "movie"
        if any(w in prompt for w in ["food", "restaurant", "eat", "cafe", "dinner", "biryani"]): return "restaurant"
        if any(w in prompt for w in ["college", "exam", "student", "class", "study", "assignment"]): return "college"
        if any(w in prompt for w in ["tech", "laptop", "phone", "gadget", "software", "app", "mobile"]): return "tech"
        if any(w in prompt for w in ["tweet", "post", "social", "reply", "comment", "instagram", "facebook"]): return "social"
        if any(w in prompt for w in ["travel", "trip", "goa", "vacation", "flight", "tour"]): return "travel"
        if any(w in prompt for w in ["shop", "buy", "dress", "order", "delivery", "sale"]): return "shopping"
        if any(w in prompt for w in ["learn", "teach", "course", "degree", "education"]): return "education"
        if any(w in prompt for w in ["support", "customer", "help", "issue", "refund", "block"]): return "support"
        return "general"
        
    @staticmethod
    def apply_style(text: str, style: str) -> str:
        """Modifies the template slightly based on the requested style."""
        if style == "formal":
            return text.replace("bro", "sir").replace("mowa", "andaru").replace("ra", "andi")
        elif style == "positive":
            return text + " Highly recommended!" if not text.endswith("!") else text
        elif style == "negative":
            return text.replace("super", "worst").replace("bagundi", "bago ledu").replace("awesome", "terrible")
        return text
        
    @staticmethod
    def generate(instruction: str, style: str = "neutral", temperature: float = 0.8) -> dict:
        """
        Simulates the model generation output cleanly.
        Temperature modifies randomness of selection.
        """
        domain = DemoGenerator.infer_domain(instruction)
        
        # Select randomly
        if temperature > 1.0:
            template = random.choice(DemoGenerator.TEMPLATES[domain])
        else:
            # lower temp restricts to the top 5
            template = random.choice(DemoGenerator.TEMPLATES[domain][:5])
            
        styled_text = DemoGenerator.apply_style(template, style)
        
        logger.info(f"[DEMO_MODE] Generated mock output for domain: {domain}")
        return {
            "generated_text": styled_text,
            "raw_prompt": instruction,
            "generation_time_sec": random.uniform(0.1, 0.4), # Simulated latency
            "token_count": len(styled_text.split()),
            "decoding_strategy": "demo_fallback",
            "generation_parameters": {"temperature": temperature}
        }
