import nltk
import numpy as np

from owlready2 import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer


def synsetsCheck(word, classes):
    synsets = wordnet.synsets(word.lower())
    for synset in synsets:
        #synset = synset.lemmas()[0].name().capitalize()
        for lemma in synset.lemmas():
            lemma = lemma.name().capitalize()
            #print(lemma)
            if lemma in classes:
                return lemma
    return False


#synsetsCheck("Depth", [])
onto = get_ontology("prototype_rasshir.owl").load()
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')


with onto:
    text = "Else well then measurement. " \
           "Else well has measurement then measurement has metrics. " \
           "Else profundity then Gamma_emission"
    textRules = nltk.sent_tokenize(text)  # разбиение на предложения


    classes = np.array([])
    for owlClass in onto.classes():
        classes = np.append(classes, owlClass.name.split(onto.name + "."))

    condition = ["Else", "For", "When", "The", "A", ""]
    operators = ["And", "Or", "Not"]
    consequence = ["Then", "Should"]
    assignment = ["Has", "Have", "Be", "Was"]

    for textRule in textRules:

        print(textRule)

        rule = Imp()
        OWLRule = ""
        lastToken = ""
        isRequiredSecondArg = False
        isFirstToken = True

        tokens = tokenizer.tokenize(textRule)
        # print(tokens)

        for token in tokens:

            token = token.capitalize()
            #token = lemmatizer.lemmatize(token).capitalize()
            #print(token)

            if token in condition:  # начало правила
                continue

            elif token in operators:  # операторы
                continue

            elif token in assignment:  # правило свойства
                isRequiredSecondArg = True
                continue

            elif token in consequence:  # следствие правила
                OWLRule = OWLRule + " -> "
                isFirstToken = True

            elif token in classes:  # поиск класса или поиск синонима
                if not isFirstToken:
                    OWLRule = OWLRule + ", "
                else:
                    isFirstToken = False

                if isRequiredSecondArg:
                    OWLRule = OWLRule + "has" + token + "(?" + lastToken + ", ?" + token + ")"
                    isRequiredSecondArg = False
                else:
                    OWLRule = OWLRule + token + "(?" + token + ")"

                lastToken = token
            elif isinstance(synsetsCheck(lemmatizer.lemmatize(token), classes), str):  # поиск синонима
                token = synsetsCheck(lemmatizer.lemmatize(token), classes)
                if not isFirstToken:
                    OWLRule = OWLRule + ", "
                else:
                    isFirstToken = False

                if isRequiredSecondArg:
                    OWLRule = OWLRule + "has" + token + "(?" + lastToken + ", ?" + token + ")"
                    isRequiredSecondArg = False
                else:
                    OWLRule = OWLRule + token + "(?" + token + ")"

                lastToken = token

            else:
                print("Некорректное предложение: " + token)
                break

        print(OWLRule)

        rule = Imp()  # Implies
        rule.set_as_rule("""hasGR(?x, ?gr), Well(?x), 
        hasDeltaPHI(?x, ?deltaphi) -> hasAnomaly(?x, true)""")


        rule.set_as_rule("""""" + OWLRule + """""")

    onto.save(file="prototype_rasshirWithRules.owl", format="rdfxml")
