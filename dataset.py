import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from itertools import islice
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from pathlib import Path

# ------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parent / 'data'
BASE_PATH = str(BASE_PATH)

# 细胞系相关路径
CELL_ID_DIR = os.path.join(BASE_PATH, 'cell2id.tsv')
CELL_FEA_DIR = os.path.join(BASE_PATH, 'cell_feat.npy')
CELL_DIR = os.path.join(BASE_PATH, 'cell_features.csv')

# ============================================================
# 【重要】处理后的数据保存路径 - 使用有足够空间的分区
# 如果 /8t 分区可用，使用它；否则使用默认路径
# ============================================================
_8T_PROCESSED_DIR = '/8t/wyt/data/processed'
if os.path.exists('/8t'):
    DATAS_DIR = _8T_PROCESSED_DIR
    os.makedirs(DATAS_DIR, exist_ok=True)
    print(f"Using /8t partition for processed data: {DATAS_DIR}")
else:
    DATAS_DIR = os.path.join(BASE_PATH, 'processed')
    print(f"Using default path for processed data: {DATAS_DIR}")

# 药物相关路径
# ============================================================
# 【重要】修改这里切换不同的协同评分数据集
# ============================================================
SYNERGY_FILENAME = 'almanac_synergy_loewe.txt'  # <-- 修改这里切换数据集
SYNERGY_FILE = os.path.join(BASE_PATH, SYNERGY_FILENAME)
# 提取数据集名称: almanac_synergy_loewe.txt -> almanac_loewe
DATASET_NAME = SYNERGY_FILENAME.replace('.txt', '').replace('_synergy', '_')

DRUG_SMILE_FILE = os.path.join(BASE_PATH, 'smiles.csv')
TARGET_FILE = os.path.join(BASE_PATH, 'drug_protein_feature.pkl')
PATHWAY_FILE = os.path.join(BASE_PATH, 'drug_pathway_feature.pkl')


# ------------------------------------------------------------
# 细胞系数据处理类
# ------------------------------------------------------------
class MyTestDataset(InMemoryDataset):
    def __init__(
            self,
            root="/tmp",
            dataset="_cell",
            xt=None,
            y=None,
            xd1=None,
            xd2=None,
            xt_feature1=None,
            xt_feature2=None,
            transform=None,
            pre_transform=None,
    ):
        """
        Initialization function: try to load existing cached data, otherwise execute process to create graph data.

        Parameter description:
        - xt: cell line ID list
        - y: label list
        - xd1: drug1 features
        - xd2: drug2 features
        - xt_feature1: cell line expression feature
        - xt_feature2: cell line fusion feature matrix
        """
        self.cell2id = self.load_cell2id(
            CELL_ID_DIR
        )  # Load the cell line index mapping table
        self.testcell = np.load(CELL_FEA_DIR)  # Load cell line expression features

        super(MyTestDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("Use existing data files")
        else:
            self.process(xt, xd1, xd2, xt_feature1, xt_feature2, y)
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("Create a new data file")

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature1(self, cellId, cell_features):
        """Find the corresponding feature vector in xt_feature1 according to cellId"""
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return None

    def load_cell2id(self, cell2id_file):
        """
        Read the cell line to index mapping table (CELL_ID_DIR)
        and find the location of the corresponding cell line in the fusion feature
        """
        cell2id = {}
        with open(cell2id_file, "r") as file:
            csv_reader = csv.reader(file, delimiter="\t")
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                cell2id[row[0]] = int(row[1])
        return cell2id

    def get_cell_feature2(self, cellId):
        """Get cell line expression characteristics based on cellId"""
        if cellId in self.cell2id:
            cell_index = self.cell2id[cellId]
            return self.testcell[cell_index]
        return None

    def get_data(self, slice):
        d = [self.data[i] for i in slice]
        return MyTestDataset(d)

    """
    Customize the process method to fit the task of cell line prediction
    Inputs:
    xt: list of encoded target (categorical or one-hot),
    xd1: drug1 features
    xd2: drug2 features
    xt_feature1: cell line feature1
    xt_feature2: cell line feature2
    y: list of labels (i.e. affinity)
    Return: PyTorch-Geometric format processed data 
    """

    def process(self, xt, xd1, xd2, xt_feature1, xt_feature2, y):
        assert len(xt) == len(y) == len(xd1) == len(xd2)
        data_list = []
        slices = [0]
        for i in range(len(xt)):
            target = xt[i]  # cell line ID
            labels = y[i]  # label
            drug1_feature = xd1[i]  # drug1 feature
            drug2_feature = xd2[i]  # drug2 feature

            # Get cell line expression characteristics 1
            cell1 = self.get_cell_feature1(target, xt_feature1)
            # Get cell line fusion PPI feature 2
            cell2 = self.get_cell_feature2(target)

            # ------------------------------------------------------------
            # ⚠️ 关键修改：检查特征，如果缺失则跳过该样本
            # ------------------------------------------------------------
            if cell1 is None or cell2 is None:
                print(f"Skipping sample {i}: Cell feature not found for target: {target}")
                continue  # ✅ 使用 continue 跳过当前循环的剩余部分，不再处理这个样本

            data = DATA.Data()

            # Processing cell features
            new_cell1 = []
            for n in cell1:
                new_cell1.append(float(n))
            data.cell1 = torch.FloatTensor(new_cell1)  # ✅ 保持一维 [D_cell1]

            if isinstance(cell2, list) and isinstance(cell2[0], np.ndarray):
                new_cell2 = np.array(cell2)
            else:
                new_cell2 = cell2

            if new_cell2.ndim > 1:
                new_cell2 = new_cell2.flatten()  # 确保是 1 维

            data.cell2 = torch.FloatTensor(new_cell2)  # ✅ 保持一维 [D_cell2]

            # Processing drug features (flatten with explicit type conversion)
            drug1_flat = np.concatenate([np.array(f, dtype=np.float32).flatten() for f in drug1_feature])
            drug2_flat = np.concatenate([np.array(f, dtype=np.float32).flatten() for f in drug2_feature])

            data.drug1 = torch.FloatTensor(drug1_flat)  # ✅ 保持一维 [D_drug]
            data.drug2 = torch.FloatTensor(drug2_flat)  # ✅ 保持一维 [D_drug]

            data.y = torch.Tensor([labels])

            data_list.append(data)  # 只有处理成功的样本才会被添加到 data_list
            # ... (剩余代码不变)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_cell_data(cellfile1, cellfile2):
    """
    Load cell line features.
    Input:
    - cellfile1: cell line simple feature file (CSV format)
    - cellfile2: cell line fusion feature file (npy format)
    """
    cell_features = []
    with open(cellfile1) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            cell_features.append(row)
    cell_features1 = np.array(cell_features)

    cell2 = np.load(cellfile2)
    cell_features2 = np.array(cell2)

    return cell_features1, cell_features2


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------

def CalculateGraphFeat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]


def FeatureExtract(drug_feature):
    drug_data = [[] for _ in range(len(drug_feature))]
    for i in range(len(drug_feature)):
        feat_mat, adj_list, _ = drug_feature.iloc[i]
        drug_data[i] = CalculateGraphFeat(feat_mat, adj_list)
    return drug_data


# ------------------------------------------------------------
# 主类：完整药物 pipeline
# ------------------------------------------------------------

class GetData():
    def __init__(self):
        self.synergyfile = SYNERGY_FILE
        self.drugsmilefile = DRUG_SMILE_FILE
        self.targetfile = TARGET_FILE
        self.pathwayfile = PATHWAY_FILE

    # --------------------------------------------------------
    # 生成 Morgan 指纹 (使用新版 MorganGenerator API)
    # --------------------------------------------------------
    def product_fps(self, data):
        from rdkit.Chem import rdFingerprintGenerator
        
        data = [x for x in data if x is not None]
        data_mols = [Chem.MolFromSmiles(s) for s in data]
        data_mols = [x for x in data_mols if x is not None]
        
        # 使用新版 MorganGenerator (radius=3, nBits=1024)
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
        data_fps = [morgan_gen.GetFingerprint(mol) for mol in data_mols]

        return data_fps

    # --------------------------------------------------------
    # 计算 Jaccard + PCA 特征
    # --------------------------------------------------------
    def feature_vector(self, feature_name, df, vector_size):

        def Jaccard(matrix):
            matrix = np.array(matrix)
            numerator = np.dot(matrix, matrix.T)
            denominator = np.dot(np.ones(matrix.shape), matrix.T) + np.dot(matrix, np.ones(matrix.T.shape)) - np.dot(
                matrix, matrix.T)
            return numerator / denominator

        all_feature = []
        for name in feature_name:
            all_feature.append(df[name])

        sim_matrix = Jaccard(np.array(all_feature))
        sim_matrix = np.asarray(sim_matrix)

        n_components = min(vector_size, sim_matrix.shape[0], 1024)
        pca = PCA(n_components=n_components)
        pca.fit(sim_matrix)
        sim_matrix = pca.transform(sim_matrix)

        return sim_matrix

    # --------------------------------------------------------
    # 生成 morgan 特征
    # --------------------------------------------------------
    def create_data(self, datatype):
        if datatype == 'morgan':
            df = pd.read_csv(self.drugsmilefile)
            smiles_list = list(df['smile'])
            drug_names = list(df['name'])
            morgan_fp = {}
            
            failed_count = 0
            for i, (name, smile) in enumerate(zip(drug_names, smiles_list)):
                try:
                    fp = self.product_fps([smile])[0]
                    morgan_fp[name] = list(map(int, fp.ToBitString()))
                except Exception as e:
                    print(f"Warning: Failed to generate Morgan fingerprint for {name}: {e}")
                    failed_count += 1
                    continue
                    
            print(f"Generated Morgan fingerprints for {len(morgan_fp)} drugs, failed for {failed_count}")
            return morgan_fp

    # --------------------------------------------------------
    # synergy score → 二分类标签
    # --------------------------------------------------------
    def get_typelabel(self, score):
        return 1 if score >= 30 else 0

    # --------------------------------------------------------
    # 构建药物样本（添加 cellname）
    # --------------------------------------------------------
    def get_feature(self, drug_feature):

        durg_dataset = {'drug_encoding': [], 'cellname': [], 'label': [], 'fold': [], 'type': []}

        with open(self.synergyfile, 'r') as f:
            header = f.readline().strip().split('\t')
            print(f"Synergy file header: {header}")
            
            # 确定列的索引
            drug1_idx = 0  # drugname1
            drug2_idx = 1  # drugname2
            cell_idx = 2   # cell_line
            score_idx = 3  # synergy
            fold_idx = 4   # fold
            
            skipped_count = 0
            processed_count = 0
            
            for line in f:
                parts = line.rstrip().split('\t')
                if len(parts) < 5:
                    continue
                    
                drug1 = parts[drug1_idx]
                drug2 = parts[drug2_idx]
                cellname = parts[cell_idx]
                score = parts[score_idx]
                fold = parts[fold_idx]
                
                # 检查药物是否在特征字典中
                if drug1 not in drug_feature:
                    print(f"Warning: Drug '{drug1}' not found in drug features")
                    skipped_count += 1
                    continue
                    
                if drug2 not in drug_feature:
                    print(f"Warning: Drug '{drug2}' not found in drug features")
                    skipped_count += 1
                    continue

                try:
                    drug1_feature = drug_feature[drug1]
                    drug2_feature = drug_feature[drug2]
                    score_val = float(score)
                    fold_val = int(float(fold))
                    
                    pair = [drug1_feature, drug2_feature]

                    durg_dataset['drug_encoding'].append(pair)
                    durg_dataset['cellname'].append(cellname)
                    durg_dataset['label'].append(score_val)
                    durg_dataset['fold'].append(fold_val)
                    durg_dataset['type'].append(self.get_typelabel(score_val))

                    # symmetric pair
                    pair_sym = [drug2_feature, drug1_feature]

                    durg_dataset['drug_encoding'].append(pair_sym)
                    durg_dataset['cellname'].append(cellname)
                    durg_dataset['label'].append(score_val)
                    durg_dataset['fold'].append(fold_val)
                    durg_dataset['type'].append(self.get_typelabel(score_val))
                    
                    processed_count += 1
                    
                except (ValueError, KeyError) as e:
                    print(f"Error processing line: {line.strip()}, Error: {e}")
                    skipped_count += 1
                    continue

        print(f"Processed {processed_count} synergy pairs, skipped {skipped_count}")
        print(f"Total samples (with symmetry): {len(durg_dataset['drug_encoding'])}")

        durg_dataset['drug_encoding'] = np.array(durg_dataset['drug_encoding'], dtype=object)
        durg_dataset['cellname'] = np.array(durg_dataset['cellname'])
        durg_dataset['label'] = np.array(durg_dataset['label'])
        durg_dataset['fold'] = np.array(durg_dataset['fold'])
        durg_dataset['type'] = np.array(durg_dataset['type'])

        return durg_dataset

    # --------------------------------------------------------
    # 划分 train / test
    # --------------------------------------------------------
    def slipt(self, drugdata, foldnum):
        test_fold = foldnum

        idx_train = np.where(drugdata['fold'] != test_fold)[0]
        idx_test = np.where(drugdata['fold'] == test_fold)[0]

        traindata = pd.DataFrame()
        testdata = pd.DataFrame()

        traindata['drug_encoding'] = drugdata['drug_encoding'][idx_train].tolist()
        traindata['cellname'] = drugdata['cellname'][idx_train].tolist()
        traindata['label'] = drugdata['label'][idx_train].tolist()
        traindata['type'] = drugdata['type'][idx_train].tolist()
        traindata['index'] = range(len(idx_train))

        testdata['drug_encoding'] = drugdata['drug_encoding'][idx_test].tolist()
        testdata['cellname'] = drugdata['cellname'][idx_test].tolist()
        testdata['label'] = drugdata['label'][idx_test].tolist()
        testdata['type'] = drugdata['type'][idx_test].tolist()
        testdata['index'] = range(len(idx_test))

        return traindata, testdata

    # --------------------------------------------------------
    # 总入口：准备所有药物特征
    # --------------------------------------------------------
    def prepare(self):

        print("Preparing drug features...")
        
        # 1. 生成 Morgan 指纹
        durg_morgan = self.create_data('morgan')
        print(f"Generated Morgan fingerprints for {len(durg_morgan)} drugs")

        # 2. 加载 drug target & pathway
        with open(self.targetfile, 'rb') as f:
            drug_target = pickle.load(f)
        with open(self.pathwayfile, 'rb') as f:
            drug_pathway = pickle.load(f)
            
        print(f"Loaded target features for {len(drug_target)} drugs")
        print(f"Loaded pathway features for {len(drug_pathway)} drugs")

        df = pd.read_csv(self.drugsmilefile)
        feature_name = df["name"].tolist()
        vector_size = len(feature_name)

        # 检查特征一致性
        morgan_drugs = set(durg_morgan.keys())
        target_drugs = set(drug_target.keys())
        pathway_drugs = set(drug_pathway.keys())
        smiles_drugs = set(feature_name)
        
        print(f"Drug counts - Morgan: {len(morgan_drugs)}, Target: {len(target_drugs)}, Pathway: {len(pathway_drugs)}, SMILES: {len(smiles_drugs)}")
        
        # 找到所有特征都有的药物
        common_drugs = morgan_drugs & target_drugs & pathway_drugs & smiles_drugs
        print(f"Drugs with all features: {len(common_drugs)}")
        
        if len(common_drugs) < len(smiles_drugs):
            missing_drugs = smiles_drugs - common_drugs
            print(f"Warning: {len(missing_drugs)} drugs missing some features: {list(missing_drugs)[:5]}...")

        # 3. Jaccard PCA - 只对有 Morgan 指纹的药物计算
        valid_feature_names = [name for name in feature_name if name in durg_morgan]
        morgan_vector = self.feature_vector(valid_feature_names, durg_morgan, len(valid_feature_names))
        
        # 确保维度为 1024
        if morgan_vector.shape[1] < 1024:
            morgan_vector = np.pad(morgan_vector, ((0, 0), (0, 1024 - morgan_vector.shape[1])))
        elif morgan_vector.shape[1] > 1024:
            morgan_vector = morgan_vector[:, :1024]
            
        print(f"Morgan vector shape after padding: {morgan_vector.shape}")

        # 4. 构建最终 drug_feature - 只包含有完整特征的药物
        drug_feature = {}
        for i, name in enumerate(valid_feature_names):
            if name in common_drugs:
                drug_feature[name] = [
                    list(durg_morgan[name]),  # Morgan FP
                    list(morgan_vector[i]),  # Jaccard PCA
                    list(drug_target[name]),  # Target
                    list(drug_pathway[name])  # Pathway
                ]

        print(f"Final drug features created for {len(drug_feature)} drugs")
        return drug_feature


# ------------------------------------------------------------
# 运行示例
# ------------------------------------------------------------
if __name__ == "__main__":
    # 细胞系数据加载
    print("Loading cell data...")
    cell_features1, cell_features2 = load_cell_data(CELL_DIR, CELL_FEA_DIR)
    print(f"Cell feature1 shape: {cell_features1.shape}")
    print(f"Cell feature2 shape: {cell_features2.shape}")

    # 药物数据处理
    print("\nLoading drug data...")
    gd = GetData()
    drug_feature = gd.prepare()
    durg_dataset = gd.get_feature(drug_feature)

    print(f"\nDrug pair samples: {len(durg_dataset['drug_encoding'])}")
    print(f"Drug feature shape: {durg_dataset['drug_encoding'].shape}")
    print(f"Label shape: {durg_dataset['label'].shape}")
    print(f"Cellname shape: {durg_dataset['cellname'].shape}")

    # 数据集划分
    print("\nSplitting dataset...")
    traindata, testdata = gd.slipt(durg_dataset, foldnum=1)
    print(f"Train samples (before filtering): {len(traindata)}")
    print(f"Test samples (before filtering): {len(testdata)}")

    # 获取可用的细胞系列表
    available_cells = set()
    for row in cell_features1:
        available_cells.add(row[0])
    print(f"\nAvailable cell lines: {len(available_cells)}")

    # 过滤训练集
    train_mask = traindata['cellname'].isin(available_cells)
    traindata = traindata[train_mask].reset_index(drop=True)
    traindata['index'] = range(len(traindata))


    # 过滤测试集
    test_mask = testdata['cellname'].isin(available_cells)
    testdata = testdata[test_mask].reset_index(drop=True)
    testdata['index'] = range(len(testdata))

    print(f"Train samples (after filtering): {len(traindata)}")
    print(f"Test samples (after filtering): {len(testdata)}")

    # 构建 PyTorch-Geometric 数据集
    print("\nBuilding PyTorch-Geometric dataset...")

    # 提取训练集数据
    train_cellnames = traindata['cellname'].tolist()
    train_labels = traindata['label'].tolist()
    train_drug_pairs = traindata['drug_encoding'].tolist()


    # 分离药物对特征
    train_drug1 = [pair[0] for pair in train_drug_pairs]
    train_drug2 = [pair[1] for pair in train_drug_pairs]

    # 创建训练集
    train_dataset = MyTestDataset(
        root=DATAS_DIR,
        dataset='train_fold1',
        xt=train_cellnames,
        y=train_labels,
        xd1=train_drug1,
        xd2=train_drug2,
        xt_feature1=cell_features1,
        xt_feature2=cell_features2
    )

    print(f"Train dataset created with {len(train_dataset)} samples")
    print(f"Sample data attributes: {train_dataset[0].keys if hasattr(train_dataset[0], 'keys') else 'N/A'}")

    # 提取测试集数据
    test_cellnames = testdata['cellname'].tolist()
    test_labels = testdata['label'].tolist()
    test_drug_pairs = testdata['drug_encoding'].tolist()

    # 分离药物对特征
    test_drug1 = [pair[0] for pair in test_drug_pairs]
    test_drug2 = [pair[1] for pair in test_drug_pairs]

    # 创建测试集
    test_dataset = MyTestDataset(
        root=DATAS_DIR,
        dataset='test_fold1',
        xt=test_cellnames,
        y=test_labels,
        xd1=test_drug1,
        xd2=test_drug2,
        xt_feature1=cell_features1,
        xt_feature2=cell_features2
    )

    print(f"Test dataset created with {len(test_dataset)} samples")
    print("\nDataset construction completed successfully!")
