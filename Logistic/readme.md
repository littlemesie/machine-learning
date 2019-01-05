#逻辑回归算法
```
Logistic回归，是一个分类而不是回归的线性模型。 
在该模型中，使用逻辑函数对描述单个试验的可能结果的概率进行建模。
```
```angular2html
class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, 
    tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, 
    random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, 
    verbose=0, warm_start=False, n_jobs=None)[source]）

penalty：str, ‘l1’ or ‘l2’, default: ‘l2’，正则化规范，其中newton-cg，sag，lbfgs只能l2

dual : bool, default: False 当n_samples > n_features.默认False

tol : float, default: 1e-4 stop条件

C : float, default: 1.0 较小的值指定更强的正则化。

max_iter : int, default: 100 迭代次数 仅适用于newton-cg，sag和lbfgs求解器。

fit_intercept : bool, default: True 

intercept_scaling : float, default 1.

class_weight : dict or ‘balanced’, default: None 权重
与{class_label：weight}形式的类相关联的权重。 如果没有给出，所有课程都应该有一个重量。 “平衡”模式使用y的值自动调整与输入数据中的类频率成反比的权重，如n_samples /（n_classes * np.bincount（y））。 请注意，如果指定了sample_weight，这些权重将与sample_weight（通过fit方法传递）相乘。

solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default: ‘liblinear’。
    liblinear：使用 coordinate descent ( 坐标下降 ) (CD) 算法
    sag：随机平均梯度下降

multi_class : str, {‘ovr’, ‘multinomial’, ‘auto’}, default: ‘ovr’
    当solver ='liblinear'时，'multinomial'不可用

random_state : int, RandomState instance or None, optional, default: None
在随机数据混洗时使用的伪随机数生成器的种子。 如果是int，则random_state是随机数生成器使用的种子; 如果是RandomState实例，则random_state是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例。 在求解器=='sag'或'liblinear'时使用

```