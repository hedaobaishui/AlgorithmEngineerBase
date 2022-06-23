<!-- TOC -->

- [1 _ __ __xx__ 比较：](#1-_-__-xx-比较)
  - [1.1 笔记位置url](#11-笔记位置url)
- [添加参数](#添加参数)
- [1 argparse包](#1-argparse包)

<!-- /TOC -->
# 1 _ __ __xx__ 比较：
## 1.1 [笔记位置url](https://www.cnblogs.com/kongk/p/8643691.html)
* _xx 单下划线开头
  
  Python中没有真正的私有属性或方法,可以在你想声明为私有的方法和属性前加上单下划线,以提示该属性和方法不应在外部调用.如果真的调用了也不会出错,但不符合规范.
* "__"双下划线
  
  这个双下划线更会造成更多混乱，但它并不是用来标识一个方法或属性是私有的，真正作用是用来避免子类覆盖其内容。 重写之后子类调用的仍然是父类的函数。
* "__xx__"前后各双下划线
  “__xx__”经常是操作符或本地函数调用的magic methods。在上面的例子中，提供了一种重写类的操作符的功能。
* __call__():
  本节再介绍 Python 类中一个非常特殊的实例方法，即 __call__()。该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
  ```
  class CLanguage:
    # 定义__call__方法
    def __call__(self,name,add):
        print("调用__call__()方法",name,add)
  clangs = CLanguage()
  clangs("C语言中文网","http://c.biancheng.net")
  ```

# 添加参数


# 1 argparse包
[参考文档](https://docs.python.org/zh-cn/3/library/argparse.html#the-add-argument-method)


```
parser = argparse.ArgumentParser()
parser.add_argument('--f',type='',default = 1,help='')
out = parser.parse_args()
```
``action=``:

* 'store_true': 触发为Ｔｒｕｅ 不触发为Ｆａｌｓｅ
```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', action='store_true')
>>> parser.add_argument('--bar', action='store_false')
>>> parser.add_argument('--baz', action='store_false')
>>> parser.parse_args('--foo --bar'.split())
Namespace(foo=True, bar=False, baz=True)
```
* 'store_const' - 存储被 const 命名参数指定的值。 'store_const' 动作通常用在选项中来指定一些标志;

``const`` 参数用于保存不从命令行中读取但被各种 ArgumentParser 动作需求的常数值.当 add_argument() 通过选项（例如 -f 或 --foo）调用并且 nargs='?' 时。这会创建一个可以跟随零个或一个命令行参数的选项。当解析命令行时，如果选项后没有参数，则将用 const 代替。

``nargs=``:
* N: （一个整数）。命令行中的 N 个参数会被聚集到一个列表中。 例如:
```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', nargs=2)
>>> parser.add_argument('bar', nargs=1)
>>> parser.parse_args('c --foo a b'.split())
Namespace(bar=['c'], foo=['a', 'b'])
```
* ?: 如果可能的话，会从命令行中消耗一个参数，并产生一个单一项。如果当前没有命令行参数，则会产生 default值。注意，对于选项，有另外的用例 - 选项字符串出现但没有跟随命令行参数，则会产生 const 值。
```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', nargs='?', const='c', default='d')
>>> parser.add_argument('bar', nargs='?', default='d')
>>> parser.parse_args(['XX', '--foo', 'YY'])
Namespace(bar='XX', foo='YY')
>>> parser.parse_args(['XX', '--foo'])
Namespace(bar='XX', foo='c')
>>> parser.parse_args([])
Namespace(bar='d', foo='d')
```
* +　和　*：>=1个参数 和 >=0个参数 

``required=`` :True \ False

``choices=``:[候选列表]、eg:['apple','banana','pear','peach']




# 2.logg