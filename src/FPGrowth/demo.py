import pyfpgrowth


transactions = [
    [1, 2, 5],
    [2, 4],
    [2, 3],
    [1, 2, 4],
    [1, 3],
    [2, 3],
    [1, 3],
    [1, 2, 3, 5],
    [1, 2, 3]]

#指定支持度阈值，得到频繁项集
patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)
print(patterns)

#指定置信度阈值，得到强关联项
rules = pyfpgrowth.generate_association_rules(patterns, 0.3)
print(rules)

a = [['苹果','啤酒','大米','鸡肉'],['苹果','啤酒','大米'],['苹果','啤酒'],['苹果','芒果'],['牛奶','啤酒','大米','鸡肉'],['牛奶','啤酒','大米'],['牛奶','啤酒'],['牛奶','芒果']]
patterns1 = pyfpgrowth.find_frequent_patterns(a, 3)
print(patterns1)
rules1 = pyfpgrowth.generate_association_rules(patterns1, 0.7)
print(rules1)

