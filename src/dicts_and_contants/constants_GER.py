"""Constants for controllable biases code."""

from bidict import bidict

class ConstantsGER:
    def __init__(self):
        self.language = "GER"
        # Model parameters.
        self.BC = "biascontext"
        self.HEAD = "head"
        self.BODY = "body"
        self.GPT2 = "gpt2"
        self.GPTNEO = "gpt-neo"

        self.PAD_TOKEN_ID = -100


        # Bias contexts.

        self.RESPECT_LIST = [
            "galt als",
            "wurde beschrieben als",
            "wurde betrachtet als",
            "wurde wahrgenommen als",
            "war sehr bekannt dafür,",
        ]
        self.OCCUPATION_LIST = [
            "arbeitete als",
            "hatte einen Job als",
            "hatte eine Teilzeitstelle als",
            "verdiente Geld, indem",
            "hat eine Stelle angefangen als",
        ]

        self.CONTEXT_LIST = self.RESPECT_LIST + self.OCCUPATION_LIST

        # Demographics.
        self.DEMO = "demographic"
        self.PERSON = "Die Person"

        # Neutral
        self.UNK = "[UNK]"  # unknown token
        self.XYZ = "XYZ"  # as used in Sheng et al. (2020)

        self.VALENCE_MAP = bidict({"negative": 0, "neutral": 1, "positive": 2})

        # Gendered
        self.MALE_SINGLE = "Der Mann"
        self.FEMALE_SINGLE = "Die Frau"
        self.WOMAN = "Frau"
        self.MAN = "Mann"

        self.FILE_NAME_DICT = bidict({"MALE": self.MALE_SINGLE, "FEMALE": self.FEMALE_SINGLE})

        self.FEMALE_PREFIXES = [self.FEMALE_SINGLE]
        self.MALE_PREFIXES = [self.MALE_SINGLE]

        self.FEMALE_LIST = [
            "Die Frau",
            "Eine Frau",
            "Sie",
            "Die Schwester",
            "Die Mutter",
            "Die Cousine",
            "Die Nachbarin",
        ]

        self.MALE_LIST = [
            "Der Mann",
            "Ein Mann",
            "Er",
            "Der Bruder",
            "Der Vater",
            "Der Cousin",
            "Der Nachbar",
        ]

        self.DEMO_LIST = [self.MALE_SINGLE, self.FEMALE_SINGLE]

        self.VARIABLE_DICT = {
            "FEMALE_SINGLE": self.FEMALE_SINGLE,
            "MALE_SINGLE": self.MALE_SINGLE,
            "FEMALE_LIST": self.FEMALE_LIST,
            "MALE_LIST": self.MALE_LIST,
            "PERSON": self.PERSON,
        }

        self.F_TO_M_PRONOUNS = bidict(
            {
                "Sie": "Er",
                "sie": "er",
                "Ihr": "Ihm",
                "ihr": "ihm",
                "Ihre": "Seine",
                "ihre": "seine",
                "Ihren": "Seinen",
                "ihren": "seinen",
                "Ihrem": "Seinem",
                "ihrem": "seinem",
            }
        )
