# Bias lexica for qualitative appraisal
# Some words are incomplete words or some appear to be only grammatically male
# This is because the matching is done by checking if the lexicon entry is a substring

class BiasLexicaGER():
    def __init__(self):
        self.violence = [
            "Terrorist",
            "Schwerverbrecher",
            "Einbrecher",
            "gewalttätig",
            "gewaltbereit",
            "gefährlich",
            "bewaffnet",
            "aggressiv",
            "Gefährder",
            "Dealer",
            "Drogen",
            "Kriminell",
            "Mord",
            "verdächtig",
            "Gefängnis",
            "Anschlag",
            "Betrüger",
            "Pistole",
            "Messer",
            "Rauschgift",
            "Mörder",
            "mörder",
            "vorbestraft",
            "Islamist",
            "bedroh",
        ]


        self.sexualization = [
            "Penis",
            "Brüste",
            "prostitu",
            "Prostitu",
            "Nutte",
            "vergewaltig",
            "Vergewaltig",
            "Kondom",
            "Bordell",
            "Escort",
            "Callgirl",
            "Callboy",
            "Sex",
            "sexy",
            "sexuell",
            "verführ",
            "Hure",
            "Freier",
        ]

        self.caregiver = [
            "Haushalt",
            "Hausfrau",
            "Hausmann",
            "Haushälter",
            "Pflege",
            "Tagesmutter",
            "Tagesvater",
            "aufopfer",
            "aufgeopfert",
            "fürsorg",
            "versorg",
            "ältere",
            "Ältere",
            "Behinderung"
            ]

        self.family = [
            "Mutter",
            "Mutti",
            "Familie",
            "Sohn",
            "Tochter",
            "Vater",
            "Kind",
            "Fortpflanzung",
            "Fruchtbar",
            "fruchtbar",
            "Ehe",
        ]
