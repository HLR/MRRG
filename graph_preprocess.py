import argparse
from multiprocessing import cpu_count
from graph_utils.convert_wiqa import convert_to_entailment
from graph_utils.tokenization_utils import tokenize_statement_file, make_word_vocab
from graph_utils.grounding import ground
from graph_utils.paths import find_paths, score_paths, prune_paths, generate_path_and_graph_from_adj
from graph_utils.graph import generate_graph, generate_adj_data_from_grounded_concepts, coo_to_normalized
from graph_utils.triples import generate_triples_from_adj

input_paths = {

    'wiqa': {
        'train': './wiqa_augment/train.jsonl',
        'dev': './wiqa_augment/dev.jsonl',
        'test': './wiqa_augment/test.jsonl',
    },
    'cpnet': {
        'csv': '/data/hlr/chenzheng/data/MRRG/data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'transe': {
        'ent': '/data/hlr/chenzheng/data/MRRG/data/transe/glove.transe.sgd.ent.npy',
        'rel': '/data/hlr/chenzheng/data/MRRG/data/transe/glove.transe.sgd.rel.npy',
    },
}

output_paths = {
    'cpnet': {
        'csv': '/data/hlr/chenzheng/data/MRRG/data/cpnet/conceptnet.en.csv',
        'vocab': '/data/hlr/chenzheng/data/MRRG/data/cpnet/concept.txt',
        'patterns': '/data/hlr/chenzheng/data/MRRG/data/cpnet/matcher_patterns.json',
        'unpruned-graph': '/data/hlr/chenzheng/data/MRRG/data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': '/data/hlr/chenzheng/data/MRRG/data/cpnet/conceptnet.en.pruned.graph',
    },
    
    'wiqa': {
        'statement': {
            'train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/statement/train.statement.jsonl',
            'dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/statement/dev.statement.jsonl',
            'test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/statement/test.statement.jsonl',
            'vocab': '/data/hlr/chenzheng/data/MRRG/data/wiqa/statement/vocab.json',
        },
        'statement-with-ans-pos': {
            'train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/statement/train.statement-with-ans-pos.jsonl',
            'dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/statement/dev.statement-with-ans-pos.jsonl',
            'test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/statement/test.statement-with-ans-pos.jsonl',
        },
        'tokenized': {
            'train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/tokenized/train.tokenized.txt',
            'dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/tokenized/dev.tokenized.txt',
            'test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/grounded/train.grounded.jsonl',
            'dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/grounded/dev.grounded.jsonl',
            'test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/train.paths.raw.jsonl',
            'raw-dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/dev.paths.raw.jsonl',
            'raw-test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/test.paths.raw.jsonl',
            'scores-train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/train.paths.scores.jsonl',
            'scores-dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/dev.paths.scores.jsonl',
            'scores-test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/test.paths.scores.jsonl',
            'pruned-train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/train.paths.pruned.jsonl',
            'pruned-dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/dev.paths.pruned.jsonl',
            'pruned-test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/test.paths.pruned.jsonl',
            'adj-train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/train.paths.adj.jsonl',
            'adj-dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/dev.paths.adj.jsonl',
            'adj-test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/graph/train.graph.jsonl',
            'dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/graph/dev.graph.jsonl',
            'test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/graph/test.graph.jsonl',
            'adj-train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/graph/train.graph.adj.pk',
            'adj-dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/graph/dev.graph.adj.pk',
            'adj-test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/graph/test.graph.adj.pk',
            'nxg-from-adj-train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': '/data/hlr/chenzheng/data/MRRG/data/wiqa/triples/train.triples.pk',
            'dev': '/data/hlr/chenzheng/data/MRRG/data/wiqa/triples/dev.triples.pk',
            'test': '/data/hlr/chenzheng/data/MRRG/data/wiqa/triples/test.triples.pk',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['wiqa'], choices=['wiqa'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'wiqa': [
            # {'func': convert_to_entailment, 'args': (input_paths['wiqa']['train'], output_paths['wiqa']['statement']['train'])},
            # {'func': convert_to_entailment, 'args': (input_paths['wiqa']['dev'], output_paths['wiqa']['statement']['dev'])},
            # {'func': convert_to_entailment, 'args': (input_paths['wiqa']['test'], output_paths['wiqa']['statement']['test'])},
            # {'func': tokenize_statement_file, 'args': (output_paths['wiqa']['statement']['train'], output_paths['wiqa']['tokenized']['train'])},
            # {'func': tokenize_statement_file, 'args': (output_paths['wiqa']['statement']['dev'], output_paths['wiqa']['tokenized']['dev'])},
            # {'func': tokenize_statement_file, 'args': (output_paths['wiqa']['statement']['test'], output_paths['wiqa']['tokenized']['test'])},
            # {'func': make_word_vocab, 'args': ((output_paths['wiqa']['statement']['train'],), output_paths['wiqa']['statement']['vocab'])},
            {'func': ground, 'args': (output_paths['wiqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['wiqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['wiqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['wiqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['wiqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['wiqa']['grounded']['test'], args.nprocs)},
            {'func': find_paths, 'args': (output_paths['wiqa']['grounded']['train'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['pruned-graph'], output_paths['wiqa']['paths']['raw-train'], args.nprocs, args.seed)},
            {'func': find_paths, 'args': (output_paths['wiqa']['grounded']['dev'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['pruned-graph'], output_paths['wiqa']['paths']['raw-dev'], args.nprocs, args.seed)},
            {'func': find_paths, 'args': (output_paths['wiqa']['grounded']['test'], output_paths['cpnet']['vocab'],
                                          output_paths['cpnet']['pruned-graph'], output_paths['wiqa']['paths']['raw-test'], args.nprocs, args.seed)},
            {'func': score_paths, 'args': (output_paths['wiqa']['paths']['raw-train'], input_paths['transe']['ent'], input_paths['transe']['rel'],
                                           output_paths['cpnet']['vocab'], output_paths['wiqa']['paths']['scores-train'], args.nprocs)},
            {'func': score_paths, 'args': (output_paths['wiqa']['paths']['raw-dev'], input_paths['transe']['ent'], input_paths['transe']['rel'],
                                           output_paths['cpnet']['vocab'], output_paths['wiqa']['paths']['scores-dev'], args.nprocs)},
            {'func': score_paths, 'args': (output_paths['wiqa']['paths']['raw-test'], input_paths['transe']['ent'], input_paths['transe']['rel'],
                                           output_paths['cpnet']['vocab'], output_paths['wiqa']['paths']['scores-test'], args.nprocs)},
            {'func': prune_paths, 'args': (output_paths['wiqa']['paths']['raw-train'], output_paths['wiqa']['paths']['scores-train'],
                                           output_paths['wiqa']['paths']['pruned-train'], args.path_prune_threshold)},
            {'func': prune_paths, 'args': (output_paths['wiqa']['paths']['raw-dev'], output_paths['wiqa']['paths']['scores-dev'],
                                           output_paths['wiqa']['paths']['pruned-dev'], args.path_prune_threshold)},
            {'func': prune_paths, 'args': (output_paths['wiqa']['paths']['raw-test'], output_paths['wiqa']['paths']['scores-test'],
                                           output_paths['wiqa']['paths']['pruned-test'], args.path_prune_threshold)},
            {'func': generate_graph, 'args': (output_paths['wiqa']['grounded']['train'], output_paths['wiqa']['paths']['pruned-train'],
                                              output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
                                              output_paths['wiqa']['graph']['train'])},
            {'func': generate_graph, 'args': (output_paths['wiqa']['grounded']['dev'], output_paths['wiqa']['paths']['pruned-dev'],
                                              output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
                                              output_paths['wiqa']['graph']['dev'])},
            {'func': generate_graph, 'args': (output_paths['wiqa']['grounded']['test'], output_paths['wiqa']['paths']['pruned-test'],
                                              output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
                                              output_paths['wiqa']['graph']['test'])},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['wiqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['wiqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['wiqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['wiqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['wiqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['wiqa']['graph']['adj-test'], args.nprocs)},
            {'func': generate_triples_from_adj, 'args': (output_paths['wiqa']['graph']['adj-train'], output_paths['wiqa']['grounded']['train'],
                                                         output_paths['cpnet']['vocab'], output_paths['wiqa']['triple']['train'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['wiqa']['graph']['adj-dev'], output_paths['wiqa']['grounded']['dev'],
                                                         output_paths['cpnet']['vocab'], output_paths['wiqa']['triple']['dev'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['wiqa']['graph']['adj-test'], output_paths['wiqa']['grounded']['test'],
                                                         output_paths['cpnet']['vocab'], output_paths['wiqa']['triple']['test'])},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['wiqa']['graph']['adj-train'], output_paths['cpnet']['pruned-graph'], output_paths['wiqa']['paths']['adj-train'], output_paths['wiqa']['graph']['nxg-from-adj-train'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['wiqa']['graph']['adj-dev'], output_paths['cpnet']['pruned-graph'], output_paths['wiqa']['paths']['adj-dev'], output_paths['wiqa']['graph']['nxg-from-adj-dev'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['wiqa']['graph']['adj-test'], output_paths['cpnet']['pruned-graph'], output_paths['wiqa']['paths']['adj-test'], output_paths['wiqa']['graph']['nxg-from-adj-test'], args.nprocs)},
        ],

    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()