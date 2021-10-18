import torch
import functools


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
        self.model, self.tokenizer = self.load_model()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def load_model(cls):
        model = torch.load(cls.model_path)
        tokenizer = torch.load(cls.tokenizer_path)
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    @staticmethod
    def post_process(horoscope):
        # Removing special tokens
        horoscope = horoscope.replace("<|startoftext|>", "")

        # Making sentences capitalized normally.
        capitalized_sentences = []
        for sentence in horoscope.split("."):
            sentence = sentence[0].capitalize() + sentence[1:]
            sentence = sentence.strip()
            capitalized_sentences.append(sentence)
        horoscope = ". ".join(capitalized_sentences)

        return horoscope

    def generate_horoscope(self, zodiac_id: int):
        if torch.cuda.is_available():
            generated = self.tokenizer(f"<|startoftext|>{self.zodiac_mapping.get(zodiac_id, 'aries')},", return_tensors="pt").input_ids.cuda()
        else:
            generated = self.tokenizer(f"<|startoftext|>{self.zodiac_mapping.get(zodiac_id, 'aries')},", return_tensors="pt").input_ids

        horoscope = self.model.generate(generated, do_sample=True, top_k=30, max_length=200, top_p=0.96, temperature=0.8, num_return_sequences=1)
        decoded = self.tokenizer.decode(horoscope, skip_special_tokens=True)
        post_processed = self.post_process(decoded)

        return post_processed
