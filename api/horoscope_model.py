import functools
import re

import torch


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        model = torch.load(self.model_path, map_location=self.device)
        tokenizer = torch.load(self.tokenizer_path, map_location=self.device)
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    @classmethod
    def post_process(cls, horoscope):
        # Removing special tokens
        horoscope = horoscope.replace("<|startoftext|>", "")
        split_horoscope = horoscope.split(".")
        # Making sentences capitalized normally.
        capitalized_sentences = []
        for i in range(len(split_horoscope)):
            sentence = split_horoscope[i].strip()
            if sentence:
                if i == 0:
                    # Users already know what zodiac they are, this is just for the model to know
                    sentence = re.sub(f"({'|'.join(cls.zodiac_mapping.values())}), ", "", sentence)
                    sentence = sentence.replace("Taurus: ", "")
                sentence = sentence[0].upper() + sentence[1:]
                capitalized_sentences.append(sentence)
        horoscope = ". ".join(capitalized_sentences) + "."

        return horoscope

    def generate_horoscope(self, zodiac_id: int):
        generated = self.tokenizer(f"<|startoftext|>{self.zodiac_mapping.get(zodiac_id, 'aries')},", return_tensors="pt").input_ids
        horoscope = self.model.generate(generated, do_sample=True, top_k=30, max_length=200, top_p=0.96, temperature=0.8, num_return_sequences=1)
        decoded = self.tokenizer.decode(horoscope[0], skip_special_tokens=True)
        post_processed = self.post_process(decoded)

        return post_processed


@functools.lru_cache(maxsize=None)
def get_model():
    return HoroscopeModel()
