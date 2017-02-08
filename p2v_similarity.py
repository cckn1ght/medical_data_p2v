from gensim.models import Word2Vec
import csv
import json


def load_se():
    with open('./sider/meddra_all_se.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        terminologies = {'_'.join(line[5].lower().split()) for line in reader}
    return terminologies


def load_indication():
    with open('./sider/meddra_all_indications.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        indications_set = set()
        for line in reader:
            indications_set.add('_'.join(line[3].lower().split()))
            indications_set.add('_'.join(line[6].lower().split()))


def extract_sim(model, terminologies_set, topn=50, similarity=0.65):
    similarity_dict = {}
    similarity_dict_sim = {}
    count = 0
    for termi in terminologies_set:
        try:
            sim_all = model.most_similar(termi, topn=topn)
            sim_with_similarity = [(word, sim)
                                   for word, sim in sim_all if sim >= similarity]
            if sim_with_similarity is not None:
                similarity_dict_sim[termi] = sim_with_similarity
            else:
                similarity_dict_sim[termi] = []
            similarity_dict[termi] = sim_all[:]
            count += 1
        except:
            pass
    print('{} terminologeis are inside vocabulary'.format(count))

    with open('./phrases/p2v_phrase_similarity.json', 'w') as f:
        json.dump(similarity_dict, f, indent=2)

    with open('./phrases/p2v_phrase_similarity>{:d}%.json'.format(int(similarity * 100)), 'w') as f:
        json.dump(similarity_dict_sim, f, indent=2)


def main():
    model = Word2Vec.load('./p2vModel/p2v_model')
    terminologies_set = load_se()
    # indications_set = load_indication()
    # all_set = terminologies_set | indications_set
    all_set = terminologies_set
    print('total terminologies: {}'.format(len(all_set)))

    extract_sim(model, terminologies_set, 100, 0.70)


if __name__ == '__main__':
    main()
