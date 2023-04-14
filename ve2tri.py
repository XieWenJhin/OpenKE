import os
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vfile", type=str)
    parser.add_argument("--efile", type=str)
    parser.add_argument("--test_edges", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    raw_edf = pd.read_csv(args.efile, index_col=0)
    test_edf = pd.read_csv(args.test_edges, index_col=0)
    edf = pd.concat([raw_edf, test_edf, test_edf]).drop_duplicates(keep=False)
    # vdf = pd.read_csv(args.vfile)
    max_elabel = 0
    with open(os.path.join(args.save_dir, 'train2id.txt'), 'w') as f:
        num_triples = edf.shape[0]
        f.write(str(num_triples))
        f.write('\n')
        for idx, row in edf.iterrows():
            f.write(str(row['source_id:int']))
            f.write(' ')
            f.write(str(row['target_id:int']))
            f.write(' ')
            f.write(str(row['label_id:int']))
            f.write('\n')
            if max_elabel < row['label_id:int']:
                max_elabel = row['label_id:int']
    
    with open(os.path.join(args.save_dir, 'test2id.txt'), 'w') as f:
        num_triples = test_edf.shape[0]
        f.write(str(num_triples))
        f.write('\n')
        for idx, row in test_edf.iterrows():
            f.write(str(row['source_id:int']))
            f.write(' ')
            f.write(str(row['target_id:int']))
            f.write(' ')
            f.write(str(row['label_id:int']))
            f.write('\n')

    # vfile = open(args.vfile)
    with open(os.path.join(args.save_dir, 'entity2id.txt'), 'w') as f:
        # num_entitys = '205783'
        num_entitys = '5101080'
        f.write(num_entitys)

    with open(os.path.join(args.save_dir, 'relation2id.txt'), 'w') as f:
        f.write(str(max_elabel + 1))
    