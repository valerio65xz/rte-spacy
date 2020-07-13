import re
import spacy
import jsonlines
import gensim.downloader as api


# Estrae i token delle frasi con i relativi PoS e dipendenti sintattici
# Per le premesse avrò più frasi, per le ipotesi solo una
def get_dependences(doc):
    final_list = []
    for sent in doc.sents:
        token_list = list()

        # For each index and word
        for i, word in enumerate(sent):

            head_idx = 0
            # If is the root
            if word.head == word:
                head_idx = 0
            else:
                # Add 1 to the index otherwise
                head_idx = doc[i].head.i + 1

            # DECOMMENTARE PER FARE SPLITTING AUTOMATICO PARSER
            # Questo per gestire quando ho diverse frasi
            # if (i + 1) == 1 and token_list:
            # final_list.append(token_list)
            # token_list = []

            processed_token = [i + 1, str(word), word.lemma_, word.pos_, word.tag_, head_idx, word.dep_]
            token_list.append(processed_token)

        final_list.append(token_list)

    return final_list


# Mi prendo ricorsivamente i dipendenti del verbo
def verb_sons(text, verb, sub_tuple, used_verbs):
    # I dipendenti li prendo sia padri che figli del verbo
    for token in text:
        if token[5] is verb[0] and token[6] in ('aux', 'PART', 'advmod', 'auxpass', 'prt', 'xcomp', 'neg', 'acomp', 'cop'):
            if token not in sub_tuple:
                sub_tuple.append(token)
                used_verbs.append(token)
                verb_sons(text, token, sub_tuple, used_verbs)
        elif token[0] is verb[5] and token[6] in ('aux', 'PART', 'advmod', 'auxpass', 'prt', 'xcomp', 'neg', 'acomp', 'cop'):
            if token not in sub_tuple:
                sub_tuple.append(token)
                used_verbs.append(token)
                verb_sons(text, token, sub_tuple, used_verbs)


# Dipendenti soggetto
def subject_sons(text, subject, sub_tuple):
    for token in text:
        if token[5] is subject[0] and token[6] in ('compound', 'appos', 'cc', 'conj', 'nummod'):
            if token not in sub_tuple:
                sub_tuple.append(token)
                other_sons(text, token, sub_tuple)


# Per l'oggetto solo dipendenti figli
def other_sons(text, object, sub_tuple):
    for token in text:
        if token[5] is object[0] and token[6] in ('obj', 'dobj', 'pobj', 'compound', 'nummod', 'amod', 'npadvmod', 'conj', 'cc', 'nmod'):
            if token not in sub_tuple:
                sub_tuple.append(token)
                other_sons(text, token, sub_tuple)


# Completo la mia tupla con gli elementi di soggetto, oggetto e preposizione
def complete_tuple(text, my_tuple, subject_number, object_number, deps_number):
    subject = []
    object = []
    first_dep = []

    actual_subj = 0
    actual_obj = 0
    actual_dep = 0

    # Stabilisco l'oggetto o il dipendente diretto

    # I vari break sono in caso per interrompere il ciclo prima, mentre il vero controllo è il "not" del flag alla fine
    # Per non fargli inserire altri oggetti o dipendenti, che vanno gestiti come tuple diverse
    # I flag full servono poiché i found mi interrompono a livello di verb element (si possono avere più dipendenze da
    # un verbo composto), mentre i full interrompono a livello di tupla
    object_full = False
    dep_full = False
    for token in text:
        object_found = False
        dep_found = False
        for verb_element in my_tuple:
            if token[5] is verb_element[0] and token[6] in (
            'obj', 'dobj', 'pobj', 'attr') and not object_found and not object_full:
                if actual_obj == object_number:
                    object.append(token)
                    object_found = True
                    object_number = 0
                    break
                else:
                    actual_obj += 1
            elif token[5] is verb_element[0] and token[6] in ('prep') and not dep_found and not dep_full:

                # Controllo se è il dipendente i-esimo che mi interessa
                if actual_dep == deps_number:
                    # print('Appendo token ', token, ' con verbo ', verb_element)
                    first_dep.append(token)
                    dep_found = True
                    actual_dep = 0
                    break
                else:
                    actual_dep += 1

        # Check sui vari flag
        if object_found:
            object_full = True
        if dep_found:
            dep_full = True
        if object_found and dep_found:
            break

    # Stabilisco i dipendenti dell'oggetto (se esiste)
    if object:
        other_sons(text, object[0], object)

    # Stabilisco i figli del dipendente diretto
    if first_dep:
        other_sons(text, first_dep[0], first_dep)

    # Stabilisco il soggetto (se il verbo è root il soggetto è il figlio, altrimenti il padre).
    # Token in my tuple mi assicura di controllare un verbo composto (is now home, is to divorce)
    # e controllare i dipendenti in tutti i suoi componenti. Se lo trovo, esco.
    for token in text:
        dep_found = False
        for verb_element in my_tuple:

            # Il primo controllo serve per memorizzare un possibile relcl di coreferenza del soggetto
            if token[0] is verb_element[5] or token[5] is verb_element[0]:
                if verb_element[6] in 'relcl':

                    # Questo controllo, anche negli altri 2 blocchi, evita di duplicare roba inutile. Inoltre controllo
                    # che non sia qualche dipendente verbale incastrato fra il soggetto e il verbo principale
                    # Non aggiungo i relcl se non sto trattando l'i-esimo soggetto
                    if token not in subject and token[6] not in ('aux', 'PART', 'advmod', 'auxpass', 'prt', 'xcomp', 'neg', 'acomp', 'cop'):
                        if actual_subj == subject_number:
                            subject.append(token)

            # Faccio break appena trovo il soggetto, la tupla avrà un solo soggetto
            if verb_element[6] in 'ROOT':
                # print('Token ', token, ' Verb ', verb_element)
                if token[5] is verb_element[0] and token[6] in ('nsubj', 'csubj', 'nsubjpass', 'npadvmod'):
                    if actual_subj == subject_number:
                        if token not in subject:
                            subject.append(token)
                        dep_found = True
                        actual_subj = 0
                        break
                    else:
                        actual_subj += 1

            else:
                if token[0] is verb_element[5] or token[5] is verb_element[0]:
                    if token[6] in ('nsubj', 'csubj', 'nsubjpass', 'npadvmod'):
                        if actual_subj == subject_number:
                            if token not in subject:
                                subject.append(token)
                            dep_found = True
                            actual_subj = 0
                            break
                        else:
                            actual_subj += 1

            # Se è un 'agent', il soggetto è colui dopo il by
            if token[5] is verb_element[0] and token[6] in ('agent'):
                agent_subject_found = 0
                agent_subject = []
                for token2 in text:
                    if token2[5] is token[0] and token2[6] in ('pobj'):
                        agent_subject.append(token2)
                        agent_subject_found += 1
                        break
                if agent_subject_found == 1:
                    if actual_subj == subject_number:
                        if token not in subject:
                            subject.append(agent_subject[0])
                        dep_found = True
                        actual_subj = 0

                        # Se trovo un soggetto AGENT, devo ricalcolare l'oggetto come soggetto della frase originale
                        # Con il del svuoto la lista object originale
                        for token3 in text:
                            if token3[0] is verb_element[5] or token3[5] is verb_element[0]:
                                if token3[6] in 'nsubjpass':
                                    if token not in subject:
                                        del object[:]
                                        object.append(token3)
                                        subject_sons(text, object[0], object)

                                        # Trasformo il verbo in forma attiva
                                        # Prendo l'indice del token del verbo e lo metto in forma attiva
                                        i = j = k = 0
                                        for token4 in my_tuple:
                                            if i < token4[0]:
                                                i = token4[0]
                                                j = k
                                            k += 1

                                        active_token = my_tuple[j].copy()
                                        del my_tuple[:]
                                        active_token[1] = active_token[2]
                                        my_tuple.append(active_token)
                                        break
                        break
                    else:
                        actual_subj += 1

        if dep_found:
            break

    # Stabilisco i dipendenti del soggetto (se esiste)
    # -1 perché sicuramente il soggetto l'ho inserito per ultimo
    if subject:

        # Controllo che il soggetto effettivo non sia un "which, that" ma un nome. Se sì, prendo solo lui lasciando
        # stare i 'relcl'
        if subject[-1][3] in ('NOUN', 'PROPN'):
            temp_subject = subject[-1]
            subject = [temp_subject]

        # Stabilisco i dipendenti dell'oggetto (se esiste)
        subject_sons(text, subject[0], subject)



    return [subject, object, first_dep]


# Definizione tuple:
# [index, token, token_lemma, part_of_speech, tag, head_index, dependency]
# [3, 'in', 'in', 'ADP', 'IN', 2, 'prep']
def create_tuples(text):
    # Gli used verbs sono per non ri aggiungere gli stessi durante i vari cicli
    used_verbs = []
    my_tuple = []
    tuples_array = []

    # Flag che mi permettono di gestire il secondo ciclo una volta trovato il verbo ROOT
    keep_going = True
    root_verb_processed = False
    root_found = False
    other_verb_found = False

    while keep_going:

        # Controllo se già ho processato root, se sì non devo più ricominciare
        if root_verb_processed:
            keep_going = False

        for token in text:

            # Controllo se devo catturare il verbo root o un altro, stando attendo che non sia un verbo già usato
            if keep_going:
                if token[6] in 'ROOT':
                    root_found = True
            else:
                if token[3] in 'VERB' and token not in used_verbs:
                    other_verb_found = True

            if root_found or other_verb_found:

                # Appendo il verbo alla mia tupla
                verb = token
                my_tuple.append(token)
                used_verbs.append(verb)

                # Mi becco i figli del verbo per i verbi composti
                verb_sons(text, verb, my_tuple, used_verbs)

                # E poi i restanti.
                # Devo creare N tuple quante sono le combinazioni di diversi soggetti, oggetti e dipendenti

                # Identificazione N. soggetti
                subject_number = 0
                for token2 in text:
                    if token2[5] is verb[0] and token2[6] in ('nsubj', 'csubj', 'nsubjpass', 'npadvmod', 'agent') and token2[3] not in 'VERB':
                        subject_number += 1

                # Se ho 0 o 1, imposto 1 così nel ciclo sarà sempre una sola istanza
                if subject_number == 0:
                    subject_number += 1

                # Identificazione N. oggetti
                object_number = 0
                for token2 in text:
                    if token2[5] is verb[0] and token2[6] in ('obj', 'dobj', 'pobj', 'attr') and token2[3] not in 'VERB':
                        object_number += 1

                if object_number == 0:
                    object_number += 1

                deps_number = 0
                for token2 in text:
                    if token2[5] is verb[0] and token2[6] in ('prep') and token2[3] not in 'VERB':
                        deps_number += 1

                if deps_number == 0:
                    deps_number += 1

                # Aggiungo tante tuple quanti sono le combinazioni soggetti-oggetti-dipendenti
                # Uso una var temp così da memorizzare la tupla prima del nuovo dipendente diretto e poterla riprocessare
                # print('Numeri ROOT ', subject_number, object_number, deps_number)
                for i in range(subject_number):
                    for j in range(object_number):
                        for k in range(deps_number):
                            my_tuple_temp = my_tuple.copy()
                            my_tuple_temp.append(complete_tuple(text, my_tuple_temp, i, j, k))
                            tuples_array.append(my_tuple_temp)

                # Azzero la tupla e faccio ripartire il ciclo se ho trovato il root
                my_tuple = []
                root_found = other_verb_found = False
                if keep_going:
                    root_verb_processed = True
                    break

    # print('A ', tuples_array)
    return build_final_representations(tuples_array)


# Costruisco la rappresentazione finale della mia tupla
def build_final_representations(tuples_array):
    premise_final_tuples = []

    # Scorro le varie tuple di verbi
    for verb_sentence in tuples_array:

        tuple_verb = []
        final_rep = ''
        sub_rep_sentence = ''

        # Prima mi prendo i verbi, li ordino e creo la frase
        for tuple_element in verb_sentence:

            # In questo modo controllo se la lista è nestata o no
            # https://stackoverflow.com/questions/24180879/python-check-if-a-list-is-nested-or-not
            if not any(isinstance(i, list) for i in tuple_element):
                tuple_verb.append(tuple_element)
            else:
                for other_elements in tuple_element:
                    sorted_rep = sorted(other_elements, key=lambda a: a[0])
                    if sorted_rep:
                        for token in sorted_rep:
                            if ',' in token[1]:
                                token[1] = token[1].replace(',', '.')
                            sub_rep_sentence = sub_rep_sentence + token[1] + ' '
                    else:
                        sub_rep_sentence += '_ '
                    sub_rep_sentence = sub_rep_sentence[:-1]
                    sub_rep_sentence += ','
                sub_rep_sentence = sub_rep_sentence[:-1]
                sub_rep_sentence += ')'

        tuple_verb = sorted(tuple_verb, key=lambda a: a[0])

        for token in tuple_verb:
            if ',' in token[1]:
                token[1] = token[1].replace(',', '.')
            final_rep = final_rep + token[1] + '_'
        final_rep = final_rep[:-1]
        final_rep += '('

        final_rep += sub_rep_sentence

        premise_final_tuples.append(final_rep)

    return premise_final_tuples


# A partire da una tupla, costruisco una frase con gli elementi della tupla separati da spazio
def get_sentence_by_tuple(my_tuple):

    verb_list = ''
    word_list = ''
    word = ''
    no_comma = True

    for char in my_tuple:

        # Nel caso del verbo, li metto dopo il soggetto
        if char == '_':
            if word:
                word += ' '
                verb_list += word
            word = ''

        elif char == '(':
            if word:
                word += ' '
                verb_list += word
            word = ''

        elif char == ' ':
            if word:
                word += ' '
                word_list += word
            word = ''

        # Se incontro la prima virgola, devo mettere i verbi dopo il soggetto, altrimenti procedo normalmente
        elif char == ',':
            if word:
                if no_comma:
                    word += ' '
                    word_list += word
                    word_list += verb_list
                    verb_list = ''
                    no_comma = False
                else:
                    word += ' '
                    word_list += word

            # Questo nel caso incontro un _ solo prima della ,
            else:
                word_list += verb_list
                verb_list = ''
            word = ''

        elif char == ')':
            if word:
                word_list += word
            break

        else:
            word += char

    # Se l'ultimo carattere è uno spazio, lo rimuovo
    if word_list[-1] == ' ':
        word_list = word_list[:-1]

    return word_list


# Calcolo del nesso causale fra tuple di ipotesi e tuple della premessa
def calculate_entailment(hypo_tuples, premise_tuples, word_vectors):

    best_similarity = -999999

    # Per ogni combinazione delle tuple dell'ipotesi e delle tuple delle premesse
    for hypo_tuple in hypo_tuples:
        for sentence_premise_tuples in premise_tuples:
            for premise_tuple in sentence_premise_tuples:

                # Costruisco la frase della tupla dell'ipotesi
                hypo_word_list = get_sentence_by_tuple(hypo_tuple)
                premise_word_list = get_sentence_by_tuple(premise_tuple)

                # Calcolo la similarità (1 - la distanza)
                similarity = 1 - word_vectors.wmdistance(hypo_word_list, premise_word_list)
                # print('Frase ipotesi "', hypo_word_list, '" Frase premessa "', premise_word_list, '" Similarità', "{:.4f}".format(similarity))
                if similarity > best_similarity:
                    best_similarity = similarity

    # Stabilisco se c'è entailment o meno
    if best_similarity > 0.68:
        return True
    else:
        return False


# Validation SET RTE
rte_valset_filename = 'E:/Università Magistrale/Tesi/RTE_files/rte_valset.jsonl'

print('Caricamento modello spaCy, l\'operazione richiederà qualche secondo.')
nlp = spacy.load("en_core_web_lg")

# Strutture utili per la computazione
premises = []
sub_premises = []
premises_array = []
sub_premises_array = []
hypo_tuple = []

# Caricamento risorsa vettoriale word2vec
print('Caricamento risorsa vettoriale. L\'operazione può impiegare un paio di minuti e occupare fino a 10GB di RAM')
word_vectors = api.load("word2vec-google-news-300")  # load pre-trained word-vectors from gensim-data

# Normalizzare mi consente di avere tutti i vettori della stessa dimensione, mitigando i difetti di similarità coseno
# e distanza euclidea
word_vectors.init_sims(replace=True)
print('Caricamento completato. Inizio calcolo nessi causali degli esempi del validation set.')

# Prelevo una linea del validation set, ogni linea sarà un esempio premessa-ipotesi
with jsonlines.open(rte_valset_filename) as json_file:

    # I-esimo esempio, conteggio di match fra entailment originale e calcolato, numero righe file
    i = 0
    matches = 0
    n = 277

    # Contatore di comodità per stampare percentuale di progresso
    perc = n / 100
    j = 10

    for line in json_file.iter():
        hypothesis = get_dependences(nlp(line['hypothesis']))

        # Splitting manuale. Ogni tanto spaCy sbagliava
        # https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
        premise = [a for a in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?])\s', line['premise']) if a]
        for sentence in premise:
            sub_premises.append(get_dependences(nlp(sentence)))
        premises.append(sub_premises)
        sub_premises = []

        # i += 1
        # if i <= 1:
        # continue

        # Prendo le tuple dell'ipotesi e delle premesse, e per ogni combinazione ne calcolo la similarità
        for hypo in hypothesis:

            # Evito frasi mal formate che il parser mi trova solo un token
            if len(hypo) > 1:

                hypo_tuple = create_tuples(hypo)
                #print('H ', hypo_tuple)

            #for token in hypo:
                #print(token)

        # Per ogni frase della premessa. Questi annidamenti mi consentono di separare ESEMPIO-FRASI-FRASE
        for premise in premises:
            for sentence in premise:
                for sub_sentence in sentence:

                    sub_premises_array.append(create_tuples(sub_sentence))
                    #print()
                    premise_tuples = create_tuples(sub_sentence)

                    #print('P ', premise_tuples)
                    #for token in sub_sentence:
                        #print(token)

            # Calcolo entailment
            label = calculate_entailment(hypo_tuple, sub_premises_array, word_vectors)

            # Conto se ho un match corretto
            original_label = False
            output_label = 'not_entailment'

            if label:
                output_label = 'entailment'

            if line['label'] in 'entailment':
                original_label = True
            if original_label == label:
                matches += 1

            premises_array.append(sub_premises_array)
            sub_premises_array = []

        # Appendo una riga CSV al file
        with open("C:/Users/Valerio/Desktop/results.csv", "a") as out:
            print('', line['idx'], ';', line['premise'], ';', premises_array, ';', line['hypothesis'], ';', hypo_tuple, ';', line['label'], ';', output_label, file=out)

        premises = []
        premises_array = []
        hypo_tuple = []

        #print()

        i += 1
        if i >= j * perc:
            print('Processato il ', j, ' percento')
            j += 10
        if i >= n:
            break

# Stampo a video e nel file il risultato
print('Numero di matches:', matches, '/', i)
accuracy = matches/i
print('Accuratezza', '{:.4f}'.format(accuracy))
with open("C:/Users/Valerio/Desktop/results.csv", "a") as out:
    print('Numero di matches:', matches, '/', i, file=out)
    print('Accuratezza', '{:.4f}'.format(accuracy), file=out)
