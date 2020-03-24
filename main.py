import argparse
import time
import pandas
import itertools


'''
used in common
'''
class Table_entry(object):
    def __init__(self,
                 sup_count,
                 first_link,
                 curr_link):
        # self.itemID = itemID
        self.sup_count = sup_count
        self.first_link = first_link
        self.curr_link = curr_link


def data_preprocess_usage(fileName):
    dataset = []
    for i in range(9):
        f = open('dataset/UNIX_usage/USER'+str(i)+'/'+fileName,'r')
        l = []
        for s in f.readlines():
            s = s.strip('\n')
            if s == '**SOF**':
                l = []
            elif s != '**EOF**':
                l.append(s)
            elif s == '**EOF**':
                if not l == []:
                    dataset.append(l)

        f.close()
    # print(dataset)
    return dataset


def data_preprocess_grocery(fileName):
    '''
    Output:
        List of transactions(list)
    '''
    df = pandas.read_csv(fileName, header=0)
    transList = df['items']
    dataset = []
    for trans in transList:
        trans_split = trans[1:-1].split(',')  # remove '{' '}'
        dataset.append(trans_split)
    # print(dataset)
    # print(len(dataset))
    # print(type(dataset))
    return dataset


def gen_1_itemsets(dataset):
    '''
    Input:
        dataset: List of transactions(list)
    Output:
        1_freq_itemsets: dict(frozenset->count)
    '''
    result = {}
    for trans in dataset:
        for item in trans:
            itemset = frozenset(set([item]))
            if itemset in result:
                result[itemset] += 1
            else:
                result[itemset] = 1
    freq_sets = {}
    for item in result:
        if result[item] >= args.min_sup:
            freq_sets[item] = result[item]
    return freq_sets


def powerset(iterable):
    '''
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    Input: list
    Output: list of sets
    '''
    s = list(iterable)
    result = []
    iter_chain = itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))
    for item in iter_chain:
        # print(item)
        # print(list(item))
        # print(set(list(item)))
        # print(frozenset(list(item)))
        # temp = set()
        # for x in item:
        # for y in x:
        # temp.add(y)
        # print(temp)
        # print(frozenset(temp))
        # print('\n')
        # if len(list(item))>1:
        # assert False
        result.append(frozenset(list(item)))
    return result


def gen_assco_rules(L, args):
    '''
    Input :
        L: list of dict(frozenset->count)
    Output:
        rules: list of tuple(3) eg(from, to, confidence)

    Leverage Apriori property:
    All nonempty subsets of a frequent itemset must also be frequent.
    '''
    length = len(L)
    result = []
    for i in range(1, length):
        for freq_set in L[i]:
            l = []
            for y in freq_set:
                l.append(y)
            subsets = powerset(l)

            for sub in subsets:
                sub1 = sub.copy()
                sub2 = set(l) - sub1

                if len(sub1) > 0 and len(sub2) > 0:
                    # print(L[2].keys())
                    # print(sub1)
                    # print(len(sub1))
                    # print(type(list(L[i].keys())[0] ) )
                    # print(type(sub1))
                    # index = frozenset(sub1)
                    # print(type(index))
                    conf = L[i][freq_set] * 1.0 / L[len(sub1)][sub1]
                    if conf > args.min_conf:
                        temp = set()
                        for x in sub:
                            temp.add(x)  # convert from frozenset to set
                        triple = (temp, sub2, conf)
                        result.append(triple)
    return result


'''
FP-growth
'''

class FPTreeNode(object):
    '''
        freq_sets: dict(frozenset->count)
    '''

    def __init__(self, name, parent, count):
        # self.left  = None
        # self.right = None
        self.children = {}  # item->FPTreeNode
        self.parent = parent
        self.next_link = None
        self.name = name
        self.count = count

    def displaySelf(self):
        print("name:{}, count:{}".format(self.name, self.count))
        if self.next_link is not None:
            print("next:{}".format(self.next_link))
        if self.parent is not None:
            print("parent:{}".format(self.parent.name))
        print("childen:{}\n".format(self.children.keys()))

    def display(self):
        print("name:{}, count:{}".format(self.name, self.count))
        if self.next_link is not None:
            print("next:{}".format(self.next_link))
        if self.parent is not None:
            print("parent:{}".format(self.parent.name))
        print("childen:{}\n".format(self.children.keys()))
        for i in self.children:
            self.children[i].display()


def insert_tree(root, trans, table):
    '''
    Input: 
        trans: list
        root : tree
    Output:
    :type trans: list
    '''
    if not trans:
        return
    trans.sort(key=lambda x: table[frozenset([x])].sup_count,
               reverse=True)
    p = frozenset([trans[0]])
    if p in root.children:
        root.children[p].count += 1
        new_child = root.children[p]
        # tail = trans[1:0]
        # if len(tail) > 0:
        #     insert_tree(root, tail, table)
    else:
        new_child = FPTreeNode(name=p, parent=root, count=1)
        root.children[p] = new_child

        if table[p].first_link == None:
            table[p].first_link = new_child

        if table[p].curr_link == None:
            table[p].curr_link = new_child
        else:
            table[p].curr_link.next_link = new_child
            table[p].curr_link = new_child
    tail = trans[1:]
    if len(tail) > 0:
        insert_tree(new_child, tail, table)


def printHeadTable(head_table):
    print("START head table!")
    for item in head_table:
        print(item)
        print(head_table[item].sup_count)
        ptr = head_table[item].first_link
        while ptr is not None:
            ptr.displaySelf()
            ptr = ptr.next_link
        print("Current link")
        head_table[item].curr_link.displaySelf()
    print("END head table!")


def terminal_num(tree):
    if len(tree.children) == 0:
        return 1
    result = 0
    for child in tree.children:
        result += terminal_num(tree.children[child])
    return result


def all_nodes(root):
    '''

    :param tree:
    :return:
        L: set of nodes(item) in tree
    '''
    L = set()
    D = {}
    L.add(root.name)
    D[root.name] = root.count
    for child in root.children:
        # assert isinstance(child.name, s et)
        l, d = all_nodes(root.children[child])
        L = L | l
        D.update(d)

    return L, D


def FP_growth(tree, table, a, patterns,args):
    '''
    :param tree:
    :param table: headtable of tree, dict(item->Table_entry)
    :param a:     common suffix, frozenset
    :param patterns: freq_patterns
    :return:
    '''
    if terminal_num(tree) == 1:  # TODO ==
        node_set, node_count = all_nodes(tree)
        s = set()
        for i in node_set:
            if i is not None:
                s = s | i
        all_comb = powerset(list(s))  # list of frozensets
        all_comb.remove(frozenset())
        # print(all_comb)
        for b in all_comb:
            p = b | a
            count = min([node_count[frozenset([item])] for item in b])
            patterns.append({frozenset(p): count})
        # print(patterns)
        # assert False
    else:
        for item in table:
            p = item | a
            patterns.append({frozenset(p): table[item].sup_count})
            # construct beta's conditional pattern base
            # and then beta's conditional FP tree Tree beta ;
            cond_base = []  # list of dict(frozenset->count)
            ptr = table[item].first_link
            # 'while' : cond_base for this item
            while ptr is not None:
                up_ptr = ptr.parent
                key = frozenset()
                while up_ptr is not None:
                    if up_ptr.name is not None:
                        key = key | up_ptr.name
                    up_ptr = up_ptr.parent
                if key is not frozenset():
                    cond_base.append({key : ptr.count})
                ptr = ptr.next_link
            # print("base:{}".format(cond_base))
            # print("list:{}".format(base_to_list(cond_base)))
            # print('\n')
            # if len(cond_base) > 2:
            #     assert False
            cond_tree, cond_table = \
                construct_FPTree(base_to_list(cond_base),args)
            if len(cond_tree.children) > 0:
                FP_growth(cond_tree,cond_table,p,patterns,args)
            # print("{}:{}\n".format(item, cond_base))

def base_to_list(cond_base):
    '''
    :param cond_base: list of dict(frozenset->count)
    :return: list of lists (dataset)
    '''
    result = []
    for dic in cond_base:
        for item in dic:
            l = []
            for x in item:
                l.append(x) # convert frozenset to list
            for i in range(dic[item]):
                result.append(l)
    return result


def construct_FPTree(dataset, args):
    '''
    Input: 
        dataset: List of transactions(list)
        args   : arguments from command line
    Output:
        tree:
        head_table
    L1: dict(frozenset->count)
    '''
    L1 = gen_1_itemsets(dataset)
    # print(L1)
    pruned_dataset = []
    for l in dataset:
        l = [x for x in l if frozenset([x]) in L1.keys()]
        pruned_dataset.append(l)

    root = FPTreeNode(name=None, parent=None, count=0)
    # construct header table
    head_table = {}  # dict(item->Table_entry)
    for item in L1:
        head_table[item] = Table_entry(
            sup_count=L1[item],
            first_link=None,
            curr_link=None)
    # head_table.sort(key = lambda x: x.value().sup_count,
    #                     reverse = True )


    for trans in pruned_dataset:
        insert_tree(root, trans, head_table)
    # printHeadTable(head_table)
    # print('\n')
    return root, head_table


'''
apriori
'''


def has_infreq_subset(c, L_k):
    '''
    Input:
        c  : set , candidate of L_{k+1}
        L_k: list of sets, frequent k-itemsets
    Output:
        boolean
    '''
    for item in c:
        tmp = c - frozenset(set([item]))  # cannot directly c.remove(item)
        if tmp not in set(L_k):
            return True
    return False


def apriori_gen(L_k, args):
    '''
    Input:
        L_k: list of sets, frequent k-itemsets
    Output:
        C_{k+1}: list of sets, candidate of (k+1)-itemsets
    '''
    candi = set()
    # print("in apriorigen")
    # print(L_k)
    for s1 in L_k:
        for s2 in L_k:
            l1 = list(s1)
            l2 = list(s2)
            l1.sort()
            l2.sort()
            if l1[:-2] == l2[:-2] and l1[-1] != l2[-1]:
                # print("l1[0:-2]:{}, l1[0:-2]:{}".format(l1[0:-2],l2[0:-2]))
                # print("l1[-1]:{}, l1[-1]:{}\n".format(l1[-1],l2[-1]))
                c = s1 | s2  # union of sets
                if args.dummy == True:
                    candi.add(c)
                else:
                    if not has_infreq_subset(c, L_k):
                        candi.add(c)
    return list(candi)


def subset(C_k, trans):
    '''
    Input:
        C_k  : list of sets
        trans: list
    Output: 
        C_t  : list of sets
    '''
    C_t = []
    set_trans = set(trans)
    for itemset in C_k:
        if itemset.issubset(set_trans):
            C_t.append(itemset)
    return C_t


def apriori(dataset, args):
    '''
    Input: 
        dataset: List of transactions(list)
        args   : arguments from command line
    Output:
        L : list of L1(frozenset->count)

    L1: dict(frozenset->count)
    C_k: list of sets
    '''
    L = []
    L1 = gen_1_itemsets(dataset)

    temp = {}
    for itemset in L1:
        if L1[itemset] >= args.min_sup:
            temp[itemset] = L1[itemset]
    L1 = temp

    L.append({})  # L[0] is useless
    L.append(L1)
    k = 2
    while len(L[k - 1]) != 0:
        C_k = apriori_gen(L[k - 1].keys(), args)
        L.append({})  # L[k]
        for trans in dataset:
            C_t = subset(C_k, trans)
            for c in C_t:
                if c in L[k]:
                    L[k][c] = L[k][c] + 1
                else:
                    L[k][c] = 1
        temp = {}
        for itemset in L[k]:
            if L[k][itemset] >= args.min_sup:
                temp[itemset] = L[k][itemset]
        L[k] = temp

        k = k + 1
    # for i in L:
    #     print(i)
    return L


def main(args):

    if args.dataset == 'grocery':
        dataset = data_preprocess_grocery('dataset/GroceryStore/Groceries.csv')
    elif args.dataset == 'usage':
        dataset = data_preprocess_usage('sanitized_all.981115184025')

    dataset = [['l1', 'l2', 'l5'],
               ['l2', 'l4'],
               ['l2', 'l3'],
               ['l1', 'l2', 'l4'],
               ['l1', 'l3'],
               ['l2', 'l3'],
               ['l1', 'l3'],
               ['l1', 'l2', 'l3', 'l5'],
               ['l1', 'l2', 'l3']]

    start = time.time()
    if args.alg == 'apriori':
        freq_itemsets = apriori(dataset, args)
        # for k in range(1, len(freq_itemsets)):
        #     for itemset in freq_itemsets[k]:
        #         print("{}:{}".format(itemset, str(freq_itemsets[k][itemset])))
        # rules = gen_assco_rules(freq_itemsets, args)
    elif args.alg == 'fpgrowth':
        tree, table = construct_FPTree(dataset, args)
        patterns = []
        FP_growth(tree, table, set(), patterns, args)
        freq_itemsets = pattern_classify(patterns)
    # print(freq_itemsets)
    # if args.dummy:
    #     for i in freq_itemsets:
    #         freq_itemsets[i] = [x for x in freq_itemsets[i] if x.value() >= args.min_sup]
    rules = gen_assco_rules(freq_itemsets, args)
    end = time.time()

    set_num = 0
    for x in freq_itemsets:
        set_num += len(x)
    rules.sort(key = lambda x: x[2], reverse=True)
    output_name = 'A_'+args.dataset+'_'+args.alg+'_sup'+str(args.min_sup)\
                  +'_conf'+str(args.min_conf)+'_dummy'+str(args.dummy)\
                  +'_set'+str(set_num)\
                  +'_rule'+str(len(rules))+'_'+str(end-start)
    with open(output_name,'w') as f:
        for r in rules:
            f.write(str(r)+'\n')
    f.close()


def pattern_classify(patterns):
    '''
    :param patterns: list of dict(frozenset->count), each dict has only one key
    :return: L : list of L1(frozenset->count), the length of key of L[x] is x
    '''
    max_length = max([len(list(p.keys())[0]) for p in patterns])
    L = []
    for i in range(max_length+1):
        L.append({})

    for p in patterns:
        L[len(list(p.keys())[0])].update(p)

    return L


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser()
    main_arg_parser.add_argument('-min_sup', type=int, default=20, help='min support count')
    main_arg_parser.add_argument('-dummy', type=bool, default=True, help='if dummy, no pruning')
    main_arg_parser.add_argument('-min_conf', type=float, default=0.1, help='min support confidence')
    main_arg_parser.add_argument('-dataset', type=str, default='grocery', help='grocery or usage')
    main_arg_parser.add_argument('-alg', type=str, default='apriori', help='apriori or fpgrowth')

    # main_arg_parser.add_argument('-')

    args = main_arg_parser.parse_args()
    main(args)
