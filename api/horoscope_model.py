import string
from functools import lru_cache
from re import sub, IGNORECASE

from torch import load, device, cuda


class HoroscopeModel:
    model_path = "pickled/horoscopes_model.pt"
    tokenizer_path = "pickled/horoscopes_tokenizer.pt"
    zodiac_mapping = {
        0: "aries",
        1: "taurus",
        2: "gemini",
        3: "cancer",
        4: "leo",
        5: "virgo",
        6: "libra",
        7: "scorpio",
        8: "saggitarius",
        9: "capricorn",
        10: "aquarius",
        11: "pisces",
    }

    def __init__(self):
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.load_model()

    @lru_cache(maxsize=None)
    def load_model(self):
        model = load(self.model_path, map_location=self.device)
        tokenizer = load(self.tokenizer_path, map_location=self.device)
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    @classmethod
    def post_process(cls, horoscope):
        # Removing special tokens
        horoscope = horoscope.replace("<|startoftext|>", "")
        split_horoscope = horoscope.split(".")
        capitalized_sentences = []
        for i in range(len(split_horoscope)):
            sentence = split_horoscope[i].strip()
            if sentence:
                if i == 0:
                    # Users already know what zodiac they are, this is only used for the model predictions
                    signs = cls.zodiac_mapping.values()
                    regex = f"({'|'.join(signs)})(, |: {0,1})"
                    sentence = sub(regex, "", sentence, flags=IGNORECASE)

                # Making sentences capitalized normally.
                sentence = sentence[0].upper() + sentence[1:]
                capitalized_sentences.append(sentence)
        horoscope = ". ".join(capitalized_sentences)

        if horoscope[-1] not in string.punctuation:
            horoscope += "."

        return horoscope

    def generate_horoscope(self, zodiac_id: int):
        generated = self.tokenizer(f"<|startoftext|>{self.zodiac_mapping.get(zodiac_id, 'aries')},", return_tensors="pt").input_ids
        horoscopes = self.model.generate(generated, do_sample=True, top_k=30, max_length=200, top_p=0.96, temperature=0.8, num_return_sequences=3)
        result = ""
        for hor in horoscopes:
            decoded = self.tokenizer.decode(hor, skip_special_tokens=True)
            post_processed = self.post_process(decoded)
            result += f"{post_processed} "

        return result


@lru_cache(maxsize=None)
def get_model():
    return HoroscopeModel()
