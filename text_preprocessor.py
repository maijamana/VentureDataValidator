import re
import spacy
import dateparser
from word2number import w2n
from pint import UnitRegistry
import inflect
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string  


NER_ACTIONS = {
    "PERSON": "keep",
    "NORP": "remove",
    "FAC": "keep",
    "ORG": "keep",
    "GPE": "keep",
    "LOC": "keep",
    "PRODUCT": "keep",
    "EVENT": "keep",
    "WORK_OF_ART": "keep",
    "LAW": "keep",
    "LANGUAGE": "remove",
    "DATE": "normalize_date",
    "TIME": "normalize_time",
    "PERCENT": "normalize_percentage",
    "MONEY": "normalize_money",
    "QUANTITY": "normalize_quantity",
    "ORDINAL": "normalize_ordinal",
    "CARDINAL": "normalize_cardinal",
}

currency_map = {
    '$': 'dollars',
    'usd': 'dollars',
    'dollar': 'dollars',
    'dollars': 'dollars',

    '£': 'pounds',
    'gbp': 'pounds',
    'pound': 'pounds',
    'pounds': 'pounds',

    '€': 'euros',
    'eur': 'euros',
    'euro': 'euros',
    'euros': 'euros',

    '¥': 'yen',
    'jpy': 'yen',
    'yen': 'yen',
}

ordinal_map = {
0: "zeroth",
1: "first",
2: "second",
3: "third",
4: "fourth",
5: "fifth",
6: "sixth",
7: "seventh",
8: "eighth",
9: "ninth",
10: "tenth",
11: "eleventh",
12: "twelfth",
13: "thirteenth",
14: "fourteenth",
15: "fifteenth",
16: "sixteenth",
17: "seventeenth",
18: "eighteenth",
19: "nineteenth"
}


class TextPreprocessor:
    """
    A text preprocessing class designed to clean, normalize,
    and prepare text data for NLP tasks.

    Functionality includes:
    - Removing structural and non-textual elements (HTML, URLs, emails, punctuation)
    - Named Entity Recognition (NER) tagging and processing
    - Normalization of dates, times, money, quantities, ordinals, cardinals, and percentages
    - Lemmatization with NER-aware exclusion
    - Stop word removal
    - Final text cleaning and formatting
    """
    def __init__(self, spacy_model_name="en_core_web_sm"):
        try:
            self.en_model_spacy = spacy.load(spacy_model_name)
        except OSError:
            print(f"SpaCy model '{spacy_model_name}' not found. Please run: python -m spacy download {spacy_model_name}")
            raise

        self.ureg = UnitRegistry()
        self.p = inflect.engine()

        try:
            nltk.data.find('corpora/stopwords')
        except:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except:
            nltk.download('punkt_tab')

        self.stop_words = set(stopwords.words('english'))

        self.currency_map = currency_map
        self.NER_ACTIONS = NER_ACTIONS
        self.ordinal_map = ordinal_map

    def remove_structural_elements(self, text: str) -> str:
        """
        Removes HTML tags, URLs, email addresses, and other common irrelevant
        structural elements from the text.
        """
        cleaned_text = re.sub(r'<[^>]+>', '', text)

        cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)
        cleaned_text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)
        cleaned_text = re.sub(r'\b[A-Za-z0-9._%+-]+(?:\.[A-Za-z]{2,})+\b', '', cleaned_text)
        cleaned_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned_text)

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text


    def extract_and_label_ner(self, text: str) -> str:
        """
        Extracts Named Entities (NER) and labels them with special <NER>...</NER> tags.
        """
        doc = self.en_model_spacy(text)

        entities = {}
        for ent in doc.ents:
            entities[ent.start_char] = (ent.end_char, ent.text, ent.label_)

        labeled_text = ""
        i = 0
        while i < len(text):
            if i in entities:
                end_char, entity_text, label = entities[i]
                labeled_text += f"<NER-{label}>{entity_text}</NER-{label}>"
                i = end_char
            else:
                labeled_text += text[i]
                i += 1
        return labeled_text


    def extract_currency(self, text: str) -> str:
        """
        Helper function to extract currency symbols/codes from text.
        """
        text_lower = text.lower()
        for symbol_or_code, name in self.currency_map.items():
            if symbol_or_code in text_lower:
                return name

        for symbol_or_code, name in self.currency_map.items():
            if re.search(r'\b' + re.escape(symbol_or_code) + r'\b', text_lower):
                return name
        return ""


    def normalize_date(self, text: str) -> str:
        """
        Normalizes dates in the text to a consistent format.
        """
        if text.isdigit():
            return text

        dt = dateparser.parse(text)
        if dt:
            return dt.strftime("%Y-%m-%d")
        return text


    def normalize_time(self, text: str) -> str:
        """
        Normalizes times in the text to a consistent format.
        """
        dt = dateparser.parse(text)
        if dt:
            return dt.strftime("%H:%M:%S")
        return text


    def normalize_percentage(self, text: str) -> str:
        """
        Normalizes percentage notations in the text.
        """
        text = text.lower().strip()

        text = text.replace('%', ' percent')
        number_part = text.replace('percent', '').strip()

        try:
            number = w2n.word_to_num(number_part)
        except:
            match = re.search(r'\d+(\.\d+)?', number_part)
            if match:
                number = float(match.group())
            else:
                return text

        return f"{number}%"

    def normalize_money(self, text: str) -> str:
        text = text.lower().strip()

        currency = self.extract_currency(text)

        cleaned_text = re.sub(r'[^\w\s]', '', text)

        number = None
        try:
            number = w2n.word_to_num(cleaned_text)
        except:
            pass

        if number is None:
            digits_only = re.sub(r'\D', '', cleaned_text)
            if digits_only.isdigit():
                number = int(digits_only)

        if number is None:
            return text

        return f"{number} {currency}".strip()


    def normalize_quantity(self, text: str) -> str:
        """
        Normalizes quantity notations (km, liters, etc.) in the text.
        """
        original_text = text.lower().strip()
        clean_text = re.sub(r'[,\']', '', original_text)

        number = None
        unit_string = ""

        number_match = re.search(r'\d+(\.\d+)?', clean_text)
        if number_match:
            number = float(number_match.group())
            unit_string = clean_text.replace(number_match.group(), '', 1).strip()
        else:
            words = clean_text.split()
            for i, word in enumerate(words):
                num_words_list = []
                for j in range(i, len(words)):
                    num_words_list.append(words[j])
                    current_phrase = ' '.join(num_words_list)
                    try:
                        temp_number = w2n.word_to_num(current_phrase)
                        number = float(temp_number)
                        unit_string = ' '.join(words[j+1:]).strip()
                        break
                    except ValueError:
                        pass
                if number is not None:
                    break

        if number is None:
            return original_text

        if not unit_string and number_match:
            remaining_words = [w for w in clean_text.split() if w not in number_match.group()]
            unit_string = ' '.join(remaining_words).strip()
        elif not unit_string and number is not None:
            if not unit_string:
                return original_text

        try:
            quantity = self.ureg.Quantity(number, unit_string)
        except Exception as e:
            return original_text

        if quantity.check('[length]'):
            quantity = quantity.to('meters')
        elif quantity.check('[volume]'):
            quantity = quantity.to('liters')
        elif quantity.check('[mass]'):
            quantity = quantity.to('kilograms')

        return f"{float(quantity.magnitude)} {quantity.units}"


    def normalize_ordinal(self, text: str) -> str:
        """
        Normalizes ordinal numbers (first, second) in the text.
        """

        ordinal_number_pattern = re.compile(r'\b(\d+)(st|nd|rd|th)\b')
        plain_number_pattern = re.compile(r'\b\d+\b')

        def handle_ordinal_number(match):
            num = int(match.group(1))
            return self.p.ordinal(self.p.number_to_words(num))

        def handle_plain_number(match):
            num = int(match.group())
            if num in self.ordinal_map:
                return self.ordinal_map[num]
            elif num % 10 == 0:
                return self.p.ordinal(self.p.number_to_words(num))
            else:
                last_digit = num % 10
                base = num - last_digit
                base_word = self.p.number_to_words(base) if base > 0 else ""
                ordinal_word = self.ordinal_map.get(last_digit, str(last_digit))
                return f"{base_word} {ordinal_word}".strip()

        text = ordinal_number_pattern.sub(handle_ordinal_number, text)
        text = plain_number_pattern.sub(handle_plain_number, text)

        return text


    def normalize_cardinal(self, text: str) -> str:
        """
        Normalizes cardinal numbers (one, two) in the text.
        """
        p = inflect.engine()
        plain_number_pattern = re.compile(r'\b\d+\b')

        def handle_plain_number(match):
            num = int(match.group())
            return p.number_to_words(num)

        text = plain_number_pattern.sub(handle_plain_number, text)
        return text


    def process_ner_tag(self, match: str) -> str:
        """
        Applies specific functions to all recognized NER tags.
        """
        ner_type = match.group(1)
        content = match.group(2)
        action = self.NER_ACTIONS.get(ner_type, "keep")
        if action == "keep":
            return f"<NER> {content} </NER>"
        elif action == "remove":
            return content
        elif action.startswith("normalize_"):
            func = getattr(self, action)
            normalized = func(content)
            return f"<NER> {normalized} </NER>"
        else:
            return content


    def process_text(self, text: str) -> str:
        """
        Calls self.process_ner_tag for further text processing.
        """
        pattern = re.compile(r"<NER-([A-Z_]+)>(.*?)</NER-\1>", re.DOTALL)
        return pattern.sub(self.process_ner_tag, text)


    def remove_non_textual_elements(self, text: str) -> str:
        """
        Helper function to remove punctuation from the text.
        """
        parts = re.split(r'(<NER>.*?</NER>)', text, flags=re.DOTALL)

        html_pattern = re.compile(r'<[^>]+>')
        url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
        email_pattern = re.compile(r'\S+@\S+')

        for i, part in enumerate(parts):
            if not part.startswith('<NER>'):
                part = html_pattern.sub('', part)
                part = url_pattern.sub('', part)
                part = email_pattern.sub('', part)
                parts[i] = part

        return ''.join(parts)


    def remove_stop_words_punctuation(self, text: str) -> str:
        """
        Lowers text. Removes stop words and punctuation from the text.
        """
        if not text:
            return ""

        punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]+")
        text = self.remove_non_textual_elements(text)
        parts = re.split(r'(<NER>.*?</NER>)', text, flags=re.DOTALL)

        for i, part in enumerate(parts):
            if not (part.startswith('<NER>') and part.endswith('</NER>')):
                lowered = part.lower()
                no_punct = punctuation_pattern.sub(' ', lowered)
                tokens = word_tokenize(no_punct)
                filtered_tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
                parts[i] = ' '.join(filtered_tokens)

        return ''.join(parts)


    def custom_lemmatizer(self, text: str) -> str:
        """
        Lemmatizes the text while preserving words inside <NER>...</NER> tags.
        """
        ner_matches = re.findall(r'<NER>(.*?)</NER>', text)

        ner_words_to_exclude = set()
        for match in ner_matches:
            ner_words_to_exclude.update(match.split())

        processed_text = re.sub(r'<NER>.*?</NER>', '', text)

        doc = self.en_model_spacy(processed_text)

        lemmatized_tokens = []
        original_tokens_map = {}

        for token in doc:
            if token.text in ner_words_to_exclude:
                lemmatized_tokens.append(token.text)
            else:
                lemmatized_tokens.append(token.lemma_)

            original_tokens_map[token.text] = token.lemma_ if token.text not in ner_words_to_exclude else token.text

        final_output_parts = []

        parts = re.split(r'(<NER>.*?</NER>)', text)

        for part in parts:
            if part.startswith('<NER>') and part.endswith('</NER>'):
                final_output_parts.append(part)
            else:
                sub_doc = self.en_model_spacy(part)
                lemmatized_sub_parts = []
                for token in sub_doc:
                    if token.text in ner_words_to_exclude:
                        lemmatized_sub_parts.append(token.text)
                    else:
                        lemmatized_sub_parts.append(token.lemma_)
                final_output_parts.append(" ".join(lemmatized_sub_parts))

        return " ".join(final_output_parts)


    def remove_ner_tags(self, text: str) -> str:
        """
        Removes the <NER> and </NER> tags from the text.
        """
        cleaned_text = re.sub(r'<NER>', '', text)
        cleaned_text = re.sub(r'</NER>', '', cleaned_text)
        return cleaned_text


    def preprocess_document(self, raw_text: str) -> str:
        """
        Executes the full text preprocessing pipeline.
        """
        cleaned_text = self.remove_structural_elements(raw_text)
        ner_labeled_text = self.extract_and_label_ner(cleaned_text)
        processed_ner_text = self.process_text(ner_labeled_text)
        lemmatized_text = self.custom_lemmatizer(processed_ner_text)
        text_after_removal = self.remove_stop_words_punctuation(lemmatized_text)
        final_text = self.remove_ner_tags(text_after_removal)

        return final_text