# CuTe 的 Layout（布局）

本文档介绍 CuTe 的核心抽象：`Layout`。
从本质上看，`Layout` 将坐标空间映射到索引空间。

`Layout` 为多维数组的访问提供统一的接口，
从而抽象掉数组元素在内存中如何组织的细节。
这样用户可以编写以通用方式访问多维数组的算法，
在 Layout 发生变化时无需修改用户代码。例如，行主序的 MxN Layout 和列主序的 MxN Layout 在软件层面可以等同处理。

CuTe 还提供了一套「`Layout` 的代数」。
`Layout` 可以组合和变换，
用于构造更复杂的 Layout，
以及在一个 Layout 之上对另一个 Layout 做分块。
这有助于用户将数据的 Layout 按线程的 Layout 进行分区。

## 基础类型与概念

### 整数

CuTe 大量使用动态整数（仅在运行时常知）和静态整数（编译时常知）。

* 动态整数（或称「运行时整数」）就是普通的整型，如 `int`、`size_t`、`uint16_t`。凡是满足 `std::is_integral<T>` 的类型，在 CuTe 中都被视为动态整数。

* 静态整数（或称「编译时整数」）是诸如 `std::integral_constant<Value>` 的实例。这类类型把值编码为 `static constexpr` 成员，并支持转为对应的动态类型，因此可以和动态整数一起参与运算。CuTe 定义了自己的 CUDA 兼容静态整型 `cute::C<Value>`，以及重载的数学运算符，使得对静态整数的运算结果仍是静态整数。CuTe 还定义了简写别名 `Int<1>`、`Int<2>`、`Int<3>` 以及 `_1`、`_2`、`_3`，在示例中会经常出现。

CuTe 尽量对静态和动态整数一视同仁。在后续示例中，所有动态整数都可以替换成静态整数，反之亦然。在 CuTe 中提到「整数」时，通常指静态或动态整数。

CuTe 提供了一系列用于整数的 traits：
* `cute::is_integral<T>`：判断 `T` 是否为静态或动态整型。
* `cute::is_std_integral<T>`：判断 `T` 是否为动态整型，等价于 `std::is_integral<T>`。
* `cute::is_static<T>`：判断 `T` 是否为空类型（实例化不依赖任何动态信息），等价于 `std::is_empty`。
* `cute::is_constant<N,T>`：判断 `T` 是否为静态整数且其值等于 `N`。

更多信息参见 [integral_constant 实现](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric/integral_constant.hpp)。

### Tuple（元组）

Tuple 是零个或多个元素的有序有限列表。
[`cute::tuple` 类](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container/tuple.hpp) 行为类似 `std::tuple`，但可在设备和主机上使用。它对模板参数施加了一些限制，并对实现做了简化以提升性能和可读性。

### IntTuple

CuTe 将 IntTuple 概念定义为：一个整数，或由 IntTuple 组成的 tuple。注意这是递归定义。
在 C++ 中，我们定义了对 [IntTuple 的操作](https://github.com/NVIDIA/cutlass/tree/main/include/cute/int_tuple.hpp)。

`IntTuple` 的例子包括：
* `int{2}`，动态整数 2。
* `Int<3>{}`，静态整数 3。
* `make_tuple(int{2}, Int<3>{})`，由动态 2 和静态 3 组成的 tuple。
* `make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})`，由动态 42、由静态 1 和动态 3 组成的 tuple，以及静态 17 所组成的 tuple。

CuTe 将 IntTuple 概念复用于多种含义，
包括 Shape（形状）、Stride（步幅）、Step 和 Coord，
详见 [`include/cute/layout.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp)。

对 `IntTuple` 定义的操作包括：

* `rank(IntTuple)`：`IntTuple` 中元素个数。单个整数的秩 (rank) 为 1，tuple 的秩为 `tuple_size`。

* `get<I>(IntTuple)`：`IntTuple` 的第 `I` 个元素，满足 `I < rank`。对单个整数，`get<0>` 即该整数本身。

* `depth(IntTuple)`：层级 `IntTuple` 的层数。单个整数的 depth 为 0，整数的 tuple 为 1，包含「整数 tuple」的 tuple 为 2，以此类推。

* `size(IntTuple)`：`IntTuple` 中所有元素的乘积。

我们用括号表示 IntTuple 的层级。例如 `6`、`(2)`、`(4,3)` 和 `(3,(6,2),8)` 都是 IntTuple。

### Shape 与 Stride

`Shape` 和 `Stride` 都是 IntTuple 概念。

### Layout

`Layout` 是 `(Shape, Stride)` 构成的 tuple。
语义上，它实现从 Shape 内任意坐标到经由 Stride 得到的索引的映射。

### Tensor

`Layout` 可以与数据（例如指针或数组）组合，形成 `Tensor`。`Layout` 生成的索引用于对迭代器进行下标访问以取出对应数据。关于 `Tensor` 的详细说明请参见 [教程中的 Tensor 部分](./03_tensor.md)。

## Layout 的创建与使用

`Layout` 是一对 IntTuple：`Shape` 和 `Stride`。前者定义 Layout 的抽象*形状*，后者定义*步幅*，将形状内的坐标映射到索引空间。

我们在 `Layout` 上定义了许多与 IntTuple 类似的操作。

* `rank(Layout)`：Layout 中的模式 (mode) 数量，等价于 Layout 的 shape 的 tuple 大小。

* `get<I>(Layout)`：Layout 的第 `I` 个子 Layout，满足 `I < rank`。

* `depth(Layout)`：Layout shape 的 depth。单个整数的 depth 为 0，整数 tuple 为 1，整数 tuple 的 tuple 为 2，以此类推。

* `shape(Layout)`：Layout 的 Shape。

* `stride(Layout)`：Layout 的 Stride。

* `size(Layout)`：Layout 函数的定义域 (domain) 大小，等价于 `size(shape(Layout))`。

* `cosize(Layout)`：Layout 函数的值域 (codomain) 大小（未必等于 range）。等价于 `A(size(A) - 1) + 1`。

### 分层访问函数

IntTuple 和 Layout 可以任意嵌套。
为方便起见，我们为上述部分函数提供了接受整数序列（而不止单个整数）的版本，
以便更轻松地访问嵌套在 IntTuple 或 Layout 内部的元素。
例如，我们支持 `get<I...>(x)`，其中 `I...` 是表示零个或多个（整型）模板参数的 C++ 参数包。这些分层访问函数包括：

* `get<I0,I1,...,IN>(x) := get<IN>(...(get<I1>(get<I0>(x)))...)`。提取 `x` 的第 `I0` 个元素的第 `I1` 个元素的 … 第 `IN` 个元素。

* `rank<I...>(x)  := rank(get<I...>(x))`。`x` 的第 `I...` 个元素的秩。

* `depth<I...>(x) := depth(get<I...>(x))`。`x` 的第 `I...` 个元素的 depth。

* `shape<I...>(x)  := shape(get<I...>(x))`。`x` 的第 `I...` 个元素的 shape。

* `size<I...>(x)  := size(get<I...>(x))`。`x` 的第 `I...` 个元素的 size。

在下面的示例中，你会看到用 `size<0>` 和 `size<1>` 来确定 layout 或 tensor 的第 0 和第 1 个模式 (mode) 的循环边界。

### 构造 Layout

`Layout` 有多种构造方式。
可以包含任意组合的编译时（静态）整数或运行时（动态）整数。

```c++
Layout s8 = make_layout(Int<8>{});
Layout d8 = make_layout(8);

Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                             make_stride(Int<12>{},Int<1>{}));
Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                               LayoutLeft{});
Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                               LayoutRight{});

Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                           make_stride(4,make_stride(2,1)));
Layout s2xh4_col = make_layout(shape(s2xh4),
                               LayoutLeft{});
```

`make_layout` 返回一个 `Layout`。
它根据函数参数推导类型，并返回具有对应模板参数的 `Layout`。
类似地，`make_shape` 和 `make_stride` 分别返回 `Shape` 和 `Stride`。
由于构造函数模板参数推导（CTAD）的限制，以及为了不必重复书写静态或动态整型，CuTe 常用这些 `make_*` 函数。

当省略 `Stride` 参数时，会以 `LayoutLeft` 为默认值，根据提供的 `Shape` 自动生成。`LayoutLeft` 标签从左到右对 Shape 做排他前缀积来构造 Stride，不考虑 Shape 的层级。可以理解为「广义列主序 Stride 生成」。`LayoutRight` 标签则从右到左对 Shape 做排他前缀积来构造 Stride，不考虑 Shape 的层级。对于 depth 为 1 的 shape，这相当于「行主序 Stride 生成」，但对分层 shape 得到的 stride 可能出乎意料。例如，上面 `s2xh4` 的 stride 也可以用 `LayoutRight` 生成。

对上述每个 layout 调用 `print` 会得到：

```
s8        :  _8:_1
d8        :  8:_1
s2xs4     :  (_2,_4):(_1,_2)
s2xd4     :  (_2,4):(_1,_2)
s2xd4_a   :  (_2,4):(_12,_1)
s2xd4_col :  (_2,4):(_1,_2)
s2xd4_row :  (_2,4):(4,_1)
s2xh4     :  (2,(2,2)):(4,(2,1))
s2xh4_col :  (2,(2,2)):(_1,(2,4))
```

`Shape:Stride` 这种写法在 Layout 中很常用。`_N` 表示静态整数，其他整数表示动态整数。可以看到 Shape 和 Stride 都可以由静态和动态整数混合组成。

另外，Shape 和 Stride 被视为*全等* (congruent)。也就是说，Shape 和 Stride 的 tuple 结构相同，Shape 中的每个整数在 Stride 中都有对应的整数。可用以下方式断言：

```cpp
static_assert(congruent(my_shape, my_stride));
```

### 使用 Layout

`Layout` 的基本用途是在 `Shape` 定义的坐标空间与 `Stride` 定义的索引空间之间做映射。例如，要用 2-D 表格打印任意秩为 2 的 layout，可以写：

```c++
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}
```

对上面的示例 layout 会得到如下输出：

```
> print2D(s2xs4)
  0    2    4    6
  1    3    5    7
> print2D(s2xd4_a)
  0    1    2    3
 12   13   14   15
> print2D(s2xh4_col)
  0    2    4    6
  1    3    5    7
> print2D(s2xh4)
  0    2    1    3
  4    6    5    7
```

可以看到这里打印了静态、动态、行主序、列主序和分层的 layout。`layout(m,n)` 将逻辑二维坐标 (m,n) 映射为一维索引。

有趣的是，`s2xh4` 既不是行主序也不是列主序。而且它有三个模式，但仍被解释为秩为 2，我们用的是二维坐标。具体来说，`s2xh4` 在第二个模式上有一个二维多模式，但我们仍能用一维坐标来访问该模式。下一节会进一步说明，这里先把思路再推广一步。我们用一维坐标，并把每个 layout 的所有模式都视为一个整体多模式。例如，下面的 `print1D` 函数：

```c++
template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
```

对上面的示例会得到：

```
> print1D(s2xs4)
  0    1    2    3    4    5    6    7
> print1D(s2xd4_a)
  0   12    1   13    2   14    3   15
> print1D(s2xh4_col)
  0    1    2    3    4    5    6    7
> print1D(s2xh4)
  0    4    2    6    1    5    3    7
```

Layout 的任意多模式（包括整个 layout 本身）都可以接受一维坐标。更多内容见后续小节。

CuTe 提供了更多用于可视化 Layout 的打印工具。`print_layout` 会生成 Layout 映射的格式化二维表格：

```text
> print_layout(s2xh4)
(2,(2,2)):(4,(2,1))
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
```

`print_latex` 会生成 LaTeX，可用 `pdflatex` 编译成同样二维表格的彩色矢量图。

### 向量 Layout

我们将秩 (rank) 为 1 的任意 `Layout` 定义为向量。
例如，layout `8:1` 可以理解为 8 个元素、索引连续的一个向量：

```
Layout:  8:1
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

类似地，layout `8:2` 可以理解为 8 个元素、索引按 2 步长的向量：

```
Layout:  8:2
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  8 10 12 14
```

按上述秩为 1 的定义，`((4,2)):((2,1))` 也被视为向量，因为其 shape 秩为 1。内层 shape 看起来像 4x2 行主序矩阵，但多一层括号表明可以把这两个模式看作一维 8 元素向量。Stride 说明前 4 个元素步长为 2，然后是这 4 个元素的 2 份副本，步长为 1。

```
Layout:  ((4,2)):((2,1))
Coord :  0  1  2  3  4  5  6  7
Index :  0  2  4  6  1  3  5  7
```

可以看到第二组 4 个元素是第一组 4 个的副本，只是多了一个步长 1。

考虑 layout `((4,2)):((1,4))`。同样是 4 个元素步长为 1，然后 2 份这样的 4 个元素步长为 4。

```
Layout:  ((4,2)):((1,4))
Coord :  0  1  2  3  4  5  6  7
Index :  0  1  2  3  4  5  6  7
```

作为从整数到整数的函数，它与 `8:1` 相同，是恒等函数。

### 矩阵示例

推广开来，我们将秩为 2 的任意 `Layout` 定义为矩阵。例如：

```
Shape :  (4,2)
Stride:  (1,4)
  0   4
  1   5
  2   6
  3   7
```

是 4x2 列主序 layout，列方向步长为 1，行方向步长为 4; 而

```
Shape :  (4,2)
Stride:  (2,1)
  0   1
  2   3
  4   5
  6   7
```

是 4x2 行主序 layout，列方向步长为 2，行方向步长为 1。主序 (majorness) 就是哪个模式具有步长 1。

与向量 layout 一样，矩阵的每个模式也可以拆成*多模式*。
这样就能表达出行主序和列主序之外的更多 layout。例如：

```
Shape:  ((2,2),2)
Stride: ((4,1),2)
  0   2
  4   6
  1   3
  5   7
```

在逻辑上也是 4x2，行方向步长为 2，但列方向是多 stride。列方向上前 2 个元素步长为 4，然后是这些元素的副本，步长为 1。因为该 layout 在逻辑上是 4x2，
与上面的列主序和行主序示例一样，
我们仍然可以用二维坐标对它进行索引。

## Layout 概念

本节介绍 `Layout` 接受的坐标集合，以及坐标映射和索引映射的计算方式。

### Layout 兼容性

若 layout A 的 shape 与 layout B 的 shape 兼容，我们称 layout A *兼容* (compatible) 于 layout B。
Shape A 与 Shape B 兼容，当且仅当：

* A 的 size 等于 B 的 size，且
* A 内的所有坐标在 B 内都是合法坐标。

例如：
* Shape 24 与 Shape 32 不兼容。
* Shape 24 与 Shape (4,6) 兼容。
* Shape (4,6) 与 Shape ((2,2),6) 兼容。
* Shape ((2,2),6) 与 Shape ((2,2),(3,2)) 兼容。
* Shape 24 与 Shape ((2,2),(3,2)) 兼容。
* Shape 24 与 Shape ((2,3),4) 兼容。
* Shape ((2,3),4) 与 Shape ((2,2),(3,2)) 不兼容。
* Shape ((2,2),(3,2)) 与 Shape ((2,3),4) 不兼容。
* Shape 24 与 Shape (24) 兼容。
* Shape (24) 与 Shape 24 不兼容。
* Shape (24) 与 Shape (4,6) 不兼容。

也就是说，*兼容* 是 Shape 上的弱偏序，满足自反、反对称和传递。

### Layout 坐标

有了上述兼容性概念，需要强调的是每个 `Layout` 可以接受多种形式的坐标。每个 `Layout` 接受与其兼容的任意 Shape 的坐标。CuTe 通过逆字典序 (colexicographic order) 提供这些坐标集合之间的映射。

因此，所有 Layout 都提供两种基本映射：

* 通过 `Shape` 将输入坐标映射到对应的自然坐标，以及
* 通过 `Stride` 将自然坐标映射到索引。

#### 坐标映射

从输入坐标到自然坐标的映射，是在 `Shape` 内应用逆字典序（从右到左读，与「字典序」从左到右读相反）。

以 shape `(3,(2,3))` 为例。该 shape 有三套坐标：一维坐标、二维坐标和自然（h 维）坐标。

|  1-D  |   2-D   |   Natural   | |  1-D  |   2-D   |       Natural   |
| ----- | ------- | ----------- |-| ----- | ------- | ----------- |
|  `0`  | `(0,0)` | `(0,(0,0))` | |  `9`  | `(0,3)` | `(0,(1,1))` |
|  `1`  | `(1,0)` | `(1,(0,0))` | | `10`  | `(1,3)` | `(1,(1,1))` |
|  `2`  | `(2,0)` | `(2,(0,0))` | | `11`  | `(2,3)` | `(2,(1,1))` |
|  `3`  | `(0,1)` | `(0,(1,0))` | | `12`  | `(0,4)` | `(0,(0,2))` |
|  `4`  | `(1,1)` | `(1,(1,0))` | | `13`  | `(1,4)` | `(1,(0,2))` |
|  `5`  | `(2,1)` | `(2,(1,0))` | | `14`  | `(2,4)` | `(2,(0,2))` |
|  `6`  | `(0,2)` | `(0,(0,1))` | | `15`  | `(0,5)` | `(0,(1,2))` |
|  `7`  | `(1,2)` | `(1,(0,1))` | | `16`  | `(1,5)` | `(1,(1,2))` |
|  `8`  | `(2,2)` | `(2,(0,1))` | | `17`  | `(2,5)` | `(2,(1,2))` |

进入 shape `(3,(2,3))` 的每个坐标都有两个*等价*坐标，所有等价坐标都映射到同一自然坐标。再强调一遍：因为上述所有坐标都是合法输入，拥有 Shape `(3,(2,3))` 的 Layout 可以像 18 个元素的一维数组那样用一维坐标，像 3x6 元素的二维矩阵那样用二维坐标，或像 3x(2x3) 元素的 h 维 tensor 那样用 h 维（自然）坐标。

前面的 1-D print 展示了 CuTe 如何用二维坐标的逆字典序来对应一维坐标。从 `i = 0` 到 `size(layout)` 遍历，并用单整数坐标 `i` 对 layout 做索引时，会按这种「广义列主序」遍历二维坐标，即使 layout 是以行主序或更复杂方式将坐标映射到索引。

函数 `cute::idx2crd(idx, shape)` 负责坐标映射。它接受 shape 内的任意坐标，并计算该 shape 下的等价自然坐标。

```cpp
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```

#### 索引映射

从自然坐标到索引的映射，是将自然坐标与 `Layout` 的 `Stride` 做内积得到。

以 layout `(3,(2,3)):(3,(12,1))` 为例。则自然坐标 `(i,(j,k))` 会得到索引 `i*3 + j*12 + k*1`。该 layout 计算的索引如下表所示，其中 `i` 用作行坐标，`(j,k)` 用作列坐标。

```
       0     1     2     3     4     5     <== 1-D col coord
     (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D col coord (j,k)
    +-----+-----+-----+-----+-----+-----+
 0  |  0  |  12 |  1  |  13 |  2  |  14 |
    +-----+-----+-----+-----+-----+-----+
 1  |  3  |  15 |  4  |  16 |  5  |  17 |
    +-----+-----+-----+-----+-----+-----+
 2  |  6  |  18 |  7  |  19 |  8  |  20 |
    +-----+-----+-----+-----+-----+-----+
```

函数 `cute::crd2idx(c, shape, stride)` 负责索引映射。它接受 shape 内的任意坐标，若尚未为自然坐标则先计算等价自然坐标，然后与 stride 做内积得到索引。

```cpp
auto shape  = Shape <_3,Shape<  _2,_3>>{};
auto stride = Stride<_3,Stride<_12,_1>>{};
print(crd2idx(   16, shape, stride));       // 17
print(crd2idx(_16{}, shape, stride));       // _17
print(crd2idx(make_coord(   1,   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},_5{}), shape, stride));  // _17
print(crd2idx(make_coord(   1,make_coord(   1,   2)), shape, stride));  // 17
print(crd2idx(make_coord(_1{},make_coord(_1{},_2{})), shape, stride));  // _17
```

## Layout 操作

### 子 Layout

可以用 `layout<I...>` 获取子 Layout：

```cpp
Layout a   = Layout<Shape<_4,Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
Layout a0  = layout<0>(a);                     // 4:1
Layout a1  = layout<1>(a);                     // (3,6):(4,12)
Layout a10 = layout<1,0>(a);                   // 3:4
Layout a11 = layout<1,1>(a);                   // 6:12
```

也可用 `select<I...>`：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = select<1,3>(a);                   // (3,7):(2,30)
Layout a01 = select<0,1,3>(a);                 // (2,3,7):(1,2,30)
Layout a2  = select<2>(a);                     // (5):(6)
```

或 `take<ModeBegin, ModeEnd>`：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = take<1,3>(a);                     // (3,5):(2,6)
Layout a14 = take<1,4>(a);                     // (3,5,7):(2,6,30)
// take<1,1> not allowed. Empty layouts not allowed.
```

### 拼接

可以给 `make_layout` 传入 `Layout` 进行包装和拼接：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout row = make_layout(a, b);                 // (3,4):(1,3)
Layout col = make_layout(b, a);                 // (4,3):(3,1)
Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))
Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
```

也可用 `append`、`prepend` 或 `replace` 进行组合：

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout ab = append(a, b);                       // (3,4):(1,3)
Layout ba = prepend(a, b);                      // (4,3):(3,1)
Layout c  = append(ab, ab);                     // (3,4,(3,4)):(1,3,(1,3))
Layout d  = replace<2>(c, b);                   // (3,4,4):(1,3,3)
```

### 分组与展平

Layout 的模式可以用 `group<ModeBegin, ModeEnd>` 分组，用 `flatten` 展平：

```cpp
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)
Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

对模式进行分组、展平和重排，可以在不移动数据的情况下将 tensor 重新解释为矩阵、将矩阵解释为向量、将向量解释为矩阵等。

### 切片

`Layout` 可以被切片，但切片更适合对 `Tensor` 进行。切片细节参见 [Tensor 部分](./03_tensor.md)。

## 小结

* `Layout` 的 `Shape` 定义其坐标空间：

    * 每个 `Layout` 都有一个一维坐标空间，
      可用于按逆字典序遍历坐标空间。

    * 每个 `Layout` 都有一个 R 维坐标空间，
      其中 R 为 layout 的秩。
      R 维坐标的逆字典序枚举
      与上面的一维坐标对应。

    * 每个 `Layout` 都有一个 h 维（自然）坐标空间，其中 h 表示「层级」。这些坐标按逆字典序排列，该顺序的枚举与上面的一维坐标对应。自然坐标与 `Shape` *全等*，即坐标的每个元素都有 Shape 中对应的元素。

* `Layout` 的 `Stride` 将坐标映射到索引：

    * 自然坐标各元素与 `Stride` 各元素的内积得到最终的索引。

对每个 `Layout`，都存在一个与其兼容的整数 `Shape`，即 `size(layout)`。因此可以总结为：

> Layout 是从整数到整数的函数。

若熟悉 C++23 的 `mdspan`，
这里与 `mdspan` layout mapping 和 CuTe `Layout` 有一个重要区别。在 CuTe 中，`Layout` 是一等公民，天然支持层级结构以自然表示行主序和列主序之外的映射，并能用层级坐标进行索引。
（`mdspan` 的 layout mapping 也可以表示层级函数，
但需要自定义 layout。）
`mdspan` 的输入坐标必须与 `mdspan` 的 shape 相同; 
多维 `mdspan` 不接受一维坐标。

## Copyright

Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
