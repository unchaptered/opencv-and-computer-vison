[< Backwards](../../README.md)

# Deep Learning

- [x] Perceptron
    - [x] 1. Single Perceptron (AN : artificate neuron)
    - [x] 2. Multiple Perceptron (ANN : artificate neuron network)
    - [x] 3. Activate - Sigmoid Functions
    - [x] 4. Activate - Hyperbolic Tangent Functions
    - [x] 5. Activate - Rectified Linear Unit(ReLU)
    - [x] 6. What is Cost Function?
    - [x] 7. Cost - Quadratic Cost
    - [x] 8. Cost - Cross Entropy
    - [x] 9. Conclusion
- [x] Gradient Descent and BackPropagation
- [x] Keras
- [x] MNIST Data Set
- [x] CNN : Convolution Neural Network (인공신경망)
    - [x] 1. Tensors
    - [x] 2. DNN vs CNN
    - [x] 3. Convolutins and Filters
    - [x] 4. Padding for Edge
    - [x] 5. What is Convolution?
    - [x] 6. What is Pooling Layers?
    - [x] 7. What is Dropout?

## Perceptron

대다수의 경우 Perceptron은 n개 이하의 요청에 1개의 결과를 반환합니다.

이러한 경우 Activate Functions의 알고리즘이 빛을 발하는데, 보편적으로 Hyperbolic Tangent Functions와 Rectified Linear Unit이 좋은 결과를 반환합니다.

1. [Perceptron, artificial neuron](./README.md#1-single-perceptron-an--artificate-neuron)
2. [Multiple Perceptron](./README.md#2-multiple-perceptron-ann--artificate-neuron-network)
3. [Activate - Sigmoid Functions](./README.md#3-activate---sigmoid-functions)
4. [Activate - Hyperbolic Tangent Functions](./README.md#4-activate---hyperbolic-tangent-functions)
5. [Activate - Rectified Linear Unit(ReLU)](./README.md#5-activate---rectified-linear-unitrelu)
6. [What is Cost Function?](./README.md#6-what-is-cost-function)
7. [Cost - Quadratic Cost](./README.md#7-cost---quadratic-cost)
8. [Cost - Cross Entropy](./README.md#8-cost---cross-entropy)
9. [Conclusion](./README.md#9-conclusion)

### 1. Single Perceptron (AN : artificate neuron)

[나무위키 / Perceptron](https://en.wikipedia.org/wiki/Artificial_neuron)은 n개 이하의 Inputs를 입력받아 Activate Functions를 실행시킵니다.

다음과 같은 Actviate Functions가 있다고 가정해보겠습니다.

AF는 모든 Inputs을 합산하여 결과값을 산출합니다.<br>
이 결과값이 양수라면 `1 혹은 Outputs 1`을 반환하고 음수라면 `0`을 리턴합니다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Neuron3.svg/400px-Neuron3.svg.png)

간혹 결과값이 0이라면 Bias(편항값)을 결과값에 더해서 이를 해결합니다.

아래 시그마 연산식이 일반적인 Activate Functions이라고 했을 때, `+b`라는 편향값을 통해서 에러를 방지할 수 있습니다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/8773ce7afe8e89c86732f0b59beb83f41f23c832)

### 2. Multiple Perceptron (ANN : artificate neuron network)

실제로는 Perceptron은 다수의 레이러 구조 하에서 작용합니다.

1. Input Layers 
    - 실제 데이터의 값
    - 데이터를 입력값으로 받아들이는 레이어
2. Hidden Layers
    - Input Layers 와 Output Layers의 사이에 존재
    - 3개 이상의 Hidden Layers가 있으면 이를 Deep Network(심층 신경망)으로 불림
3. Output Layer
    - 출력값을 산출하는 레이어

![](https://static.javatpoint.com/tutorial/tensorflow/images/multi-layer-perceptron-in-tensorflow.png)

더 많은 레이어를 거칠수록 추상화 레벨이 올라가게 됩니다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/be21980cc9e55ea0880327b9d4797f2a0da6d06e)

### 3. Activate - Sigmoid Functions

[#Perceptron (AN : artificate neuron)](./README.md#1-single-perceptron-an--artificate-neuron)에서 다룬 기본적인 Activate Functions의 고유한 한계는 Inputs의 총합이 -1 이거나 -1000이거나 `항상 0을 반환`하는 것에 있습니다.

따라서, 다음과 같은 [나무위키 / Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function)을 통해서 미세값이 반영된 Outputs을 얻을 수 있습니다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/faaa0c014ae28ac67db5c49b3f3e8b08415a3f2b)

이를 그래프로 보면 다음과 같습니다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Error_Function.svg/320px-Error_Function.svg.png)

### 4. Activate - Hyperbolic Tangent Functions

위에서 언급한 [#Sigmoid Functions](./README.md#3-activate---sigmoid-functions) 대신에 Hyperbolic Tangent Functions을 사용할 수 있습니다. 

이전까지는 1, 0을 반환하였지만 Sigmoid는 `1과 -1을 반환`합니다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/c5339e8f573e8aa4082e7395089fb620a5ed3de1)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/cc9e4dd5d9c44875bd6dde6356b223e5cf18880c)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/c6b439d4cd800e1c60c54e5d902c7fe5acb48302)

이를 그래프로 보면 다음과 같습니다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Sinh_cosh_tanh.svg/220px-Sinh_cosh_tanh.svg.png)

### 5. Activate - Rectified Linear Unit(ReLU)

매우 일반적으로 쓰이는 사례이며, `max(O, z)` 을 통해서 연산처리 됩니다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/ReLU_and_GELU.svg/220px-ReLU_and_GELU.svg.png)

### 6. What is Cost Function?

Perceptron의 기능을 어떻게 평가하는지 알아보기 위해서, Cost Function을 사용해서 `예상값과 결과값이 얼마나 떨어져 있는지 확인`할 생각입니다.

사용되는 변수는 다음과 같습니다.

```bash
y = 실제값
a = Perceptron의 예상값
```

가중치 시간과 편향 관점에서...

```bash
w * x + b = z

# w * x = 입력값의 가중치 시간
# b = bias 편향값
```

위와 같이 연산처리된 z는 Activate Functions에 인자로 전달이 됩니다. 즉, Sigmoid Functions를 채택한 경우에는 시그마(z)는 a라는 결과값(에상값)이 될 것입니다.

```bash
σ(z) = a
# 시그마(z) = a
```

이렇게 계산된 Perceptron 예상값 a와 실제 결곽값 y를 비교하게 됩니다.

### 7. Cost - Quadratic Cost

Quadratic Cost Function은 2차 비용함수로 공식을 적용하는 과정에서 `성능 저하`가 약간 발생할 수 있습니다.

```bash
C = Σ (y - a)^2 / n

# y = 실제값
# a = 예상값
# n = 횟수
```

[PDF / "Proper Quadratic Cost Functions with an application to AT&T"](https://flora.insead.edu/fichiersti_wp/Inseadwp1988/88-22.pdf)

### 8. Cost - Cross Entropy

Cross Entropy Functions(교차 앤트로피 함수)는 더 빠른 학습이 가능하게 만들어줍니다.

y와 `a`의 가장 큰 차이점은 교차 엔트로피 비용 함수 덕분에 Perceptron의 학습속도가 매우 빠르다는 것입니다. 


```bash
C = (-1/n) * Σ(y*ln(a) + (1-y)*ln(1-a))
```

### 9. Conclusion

Multiple Percentron(ANN)이 효율적으로 정상 작동하기 위해서는 Activate Functions, Cost Functions을 잘 선택해야 합니다.

## Gradient Descent and Backpropagation

한글 제목 : 경사 하강법과 역전파

ANN을 이용하는 경우 다음과 같은 이슈를 맞이하게 됩니다.

1. 비용 최소화점 탐색
2. 오차 조절

이러한 경우에 각각 Gradient Descent와 Backpropagation을 사용할 수 있습니다.

1. [Gradient Descent](./README.md#1-gradient-descent경사-하강법)
2. [Backpropagation](./README.md#2-backpropagation역전파)

### 1. Gradient Descent(경사 하강법)

[Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)는 최솟값을 찾는데 최적 알고리즘입니다. 따라서, Cost Functions 최적화를 원하는 개발자들에게 큰 이점을 가지고 있습니다. 

```bash
# x 축 가중치(weight)
# y 축 비용(cost)
```

비용 최소화점을 찾기 위해서 가중치가 속한 부분을 `미분`하여 기울기를 연산합니다. 이 기울기가 `최솟값(0)`이 되는 방향까지 `가중치를 조절`하여 비용 최소화점을 찾을 수 있습니다.

![](https://static.javatpoint.com/tutorial/machine-learning/images/gradient-descent-in-machine-learning1.png)

최솟값 자체를 찾는 방식은 매우 간결하고 효율적이지만, 복잡한 그래프를 구성하고 있는 경우에는 다소 난해할 수 있습니다.

### 2. Backpropagation#역전파

Gradient Descent의 방법은 모든 가중치를 수정 및 조정 하는 경우에는 적합하지 않을 수 있습니다.

이러한 경우, Backpropagtion을 사용하여 연산이 종료된 outputs의 오차를 조절하게 됩니다.

단어 그대로, Percentron Network(ANN)을 거슬러 올라가면서 오류를 계산하는 방식입니다.
각 계층마다 Comparison(비교)를 하여 단계별로 조금씩 오류를 계산합니다.
## MNIST Data Set

<details>
    <summary>Read by English</summary>
The MNIST data set contains handwritten single digits from 0 to 9. <br>
A single digit image can be represented as an array. <br>
Specifically, 28 by 28 pixels. <br>
The valeus represent the grayscale image. <br>
We can think of the entire group of the 60,000 images as a 4-demensional array. <br>
60,000 images of 1 channel 28 by 28 pixels.<br><br>
For the labels we'll use One-Hot Encoding. <br>
This means that instead of having labels such as "One", "Two", etc... <br>
We'll have a single array for each image.<br><br>
This means if the original labels of the images are given as a list of number
    - [5, 0, 4, ... 5, 6, 8]<br><br>
We will convert them to one-hot encoding (easily done with Keras)<br><br>
The label is represented based off the index position in the label array. <br>
The corresponding label will be a 1 at the index location and zero everywhere else. <br>
For example, a drawn digit of 4 would have this label array:
    - [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]<br><br>
- As a result, the labels for the training data ends up being a large 2-d array (60000, 10):
</details>

CNN과 Machine Laerning 분야에서 Data Set은 매우 일반적인 개념입니다. <br>
어느 정도 정제되어 있는 Data Set은 해당 분야에서 매우 유용하기에 꼭 그 사용방법을 알아두면 좋습니다.

Keras 안에는 다음과 같이 구성된 MNIST Data Set이 존재합니다.

- 학습 이미지 6만 개
- 세트 단위로 1만 개

![](../../images/MNIST_DATASET.png)

MNIST Data Set은 사람들이 0부터 9까지를 표기하는 다양한 문체가 드러나 있습니다. 이 Data Set의 대표적인 특징점은 다음과 같습니다.

- Format : 모든 이미지는 28 x 28 픽셀로 이루어져 있습니다.
- Grayscale : 모든 이미지는 1 채널로 이루어져 있습니다.
- Standardization : 픽셀의 값은 0~255가 아닌 0~1로 표기됩니다.

따라서 단일 이미지는 `(28, 28, 1)`, 전체 이미지는 `(60000, 28, 28, 1)`로 이루어져 있습니다.

Data Set을 다루기 위한 Data Label을 효율적으로 다루기 위해서 [One-Hot Encoding](https://wikidocs.net/22647)을 사용할 것입니다.

> <details>
>     <summary>설명 보기</summary>
> 
> One-Hot Encoding을 위한 단어 집합화를 진행하면 하나의 숫자는 길이 10의 배열을 가지게 됩니다.
> 
> 만약 숫자가 3이라면 다음과 같은 배열을 가질 것입니다.
> ```python
> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
> ```
> 
> 즉, 이미지 3장이 각각 [3, 5, 7]일 경우에 다음과 같은 배열을 가질 것입니다.
> ```python
> [
>   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
>   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
>   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
> ]
> ```

이러한 인코딩을 사용하는 이유는 Multiple Perceptron's Activet Funtion의 일종인 `Sigmoid Function`이 **길이 10의 0과 1을 다루는 자료구조 에 특화되어 있기 때문**입니다.

> </details>

## CNN : Convolution Neural Network (인공신경망)

<details>
    <summary>Read by English</summary>
We just created a Neural Netowrk for already defined features. <br>
But what if we have the raw image data? <br>
We need to learn about Convolutional Neural Netowrks in order to effctively solve the problems that image data can present!<br><br>
Just like the simple perceptron, CNNs also have their origins in biological research. <br>
Hubel and Wiesel sutdied the structure of the visual cortext in mammals, winning a Nobel Prize in 1981<br><br>
Their research revealed that neurons in the visual cortext had a small local receptive field<br><br>
This idea then inspired an ANN architecture that would become CNN
Famously implemented in the 1998 paper by Yann LeCun et al. <br>
The LeNet-5 architecture was first used to classify the MNIST data set.<br><br>
When learning about CNNs you'll often see a diagram like this: <br>
Let's break down the various aspects of a CNN seen here:<br><br>
We talked about right these.<br><br>
1. Tensors
2. DNN vs CNN
3. Convolutions and Filters
4. Padding
5. Pooling Layers
6. Review Dropout
</details>

CNN은 매우 좋지만 다른 Simple Multiple Perceptron Model과 별로 다르지 않습니다.

Multiple Perceptron Model은 기본적으로 생물학적인 모델을 기준으로 설계 & 구현되었습니다.
CNN 또한 같은 개념을 적용해서 설계 & 구현되었기 때문에, 그 복잡도만 다를 뿐 원리는 유사합니다.

1981년에 노벨상을 받은 Hubel & Wiesel의 연구의 주요소 중 하나는 다음과 같았습니다.

- 단일 뉴런이 시각 정보의 극히 일부분만을 처리한다.

이러한 국부적 세부 영역이 결합되어 시각 영역보다 큰 이미지를 생성할 수 있게 만들어줍니다. 또한 ...

- 시각 피질 안의 특정 뉴런들이 특정한 것을 감지할 때만 활성화된다.
- 뉴런들은 수직-수평선, 검은 원, 기타 등등을 감지할 때 활발할 수 있다.

ANN, CNN architecture 등에 큰 도움이 되었습니다.

![](../../images/CNN_architecture.jpeg)

이를 제대로 알기 위해서는 다음과 같은 개념들을 잘 알아야 합니다.

1. [Tensors](./README.md#1-tensors)
2. [DNN vs CNN](./README.md#2-dnn-vs-cnn)
3. [Convolutins and Filters](./README.md#3-convolutins-and-filters)
4. [Padding for Edge](./README.md#4-padding-for-edge)
5. [What is Convolution?](./README.md#5-what-is-convolution)
6. [What is Pooling Layers?](./README.md#6-what-is-pooling-layers)
7. [What is Dropout?](./README.md#7-what-is-dropout)

### 1. Tensors

| Name   | Format                                           |
| ------ | ------------------------------------------------ |
| Saclar | 3                                                |
| Vector | [3, 4, 5]                                        |
| Matrix | [[3,4], [5,6], [7,8]]                            |
| Tensor | [[[3,4], [5,6], [7,8]], [[3,4], [5,6], [7,8]]]   |

Tensors make it very convenient to feed in sets of images into our model `IHWC`.

```bash
# I : Images
# H : Heights of iamge in pixels
# W : Width of image in pixels
# C : Color channels 1-Grayscale, 3-RGB
```

### 2. DNN vs CNN

시각처리와 관련된 복합 알고리즘은 다음과 같습니다.

- 심층 신경망 - DNN : Densly Connected Neural Network
- 컨볼루션 신경망 - CNN : Convolutional Neural Network

이 챕터에서는 CNN을 쓰고 있는데 그 이유는 무엇일까요?<br>

그 해답은 `행렬곱 횟수`에 있습니다.<br>
MNIST Data Set에서는 (28, 28) 행렬을 다루고 있으므로 `784번`에 해당합니다.<br>
대다수 이미지는 최소 (256, 256) 이므로 `65,536번`에 해당합니다.

이는 기하급수적으로 증가하여 FHD 화질의 경우 (1920, 1080) 행렬로 `2,073,600번` 연산해야 합니다.<br>
이러한 경우, DNN보다 CNN이 훨씬 성능 적합한 방법이라고 할 수 있을 것입니다.

이 [tf.estimator API](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)를 사용해서 CNN을 손쉽게 구성할 수 있습니다.

### 3. Convolutins and Filters

이미지 그 자체인 Input Layers에서는 특정한 영역을 하나의 조직으로 그룹화를 진행합니다.

이렇게 기준점에 도달할 때까지 Convolutional Layer를 하나씩 쌓아가게 됩니다.

- [Quora - Why do we use convolutional layers?](https://www.quora.com/Why-do-we-use-convolutional-layers)

![](https://qph.cf2.quoracdn.net/main-qimg-5bfbe970b2bf646781b66e9e78a2214b-lq)

### 4. Padding for Edge

그러나 이미지의 Edge에는 정상적인 Convolutional Layer나 형성될 수 없습니다.

따라서, 다음과 같이 0으로 채워진 Padding Area를 편성함으로써 이를 해결할 수 있습니다.

- [Weights & Biases - Introduction to Convolutional Neural Networks with Weights & Biases](https://wandb.ai/site/articles/intro-to-cnns-with-wandb)
![Weights & Biases - Introduction to Convolutional Neural Networks with Weights & Biases](https://assets.website-files.com/5ac6b7f2924c652fd013a891/5dd6bf8fed1853d1719ec81a_s_92F7A2BE132D5E4492B0E3FF3430FFF0FB2390A4135C0D77582A2D21A2EF8567_1573598824588_Screenshot%2B2019-11-12%2B14.46.40.png)

### 5. What is Convolution?

- [11-02 자연어 처리를 위한 1D CNN(1D Convolutional Neural Networks)
](https://wikidocs.net/80437)

Filter
Stride


- [Code Craft - Creating a densely connected Neural Network](https://codecraft.tv/courses/tensorflowjs/neural-networks/creating-a-densly-connected-neural-network/)

![Creating a densely connected Neural Network](https://codecraft.tv/courses/tensorflowjs/neural-networks/creating-a-densly-connected-neural-network/img/exports/5.mnist.003.jpeg)

- [Medium - Intuitively Understanding Convolutions for Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)

### 6. What is Pooling Layers?

Poooling Layers는 입력값에 대한 복사본을 생성합니다. <br>
이를 통해서 컴퓨터의 메모리 사용량과 매개변수의 수를 감소시킵니다. <br>

Maximum Value가 커널의 최댓값이 됩니다.
아래에 길이가 (m, n)의 행렬이 있다고 할 때, (2, 2) 커널의 최댓값만 Pooling Layer에 기록하게 됩니다.

```python
# Inputs
[[0, .2, ...], [.4, .3, ...]]

# Pooling Layer (calc maximum value)
[0.4]
```

CNN은 Stride가 한칸씩 이동하면서 연산하므로 결과적으로 약 `(int(m / 2), int(n / 2))` 크기의 행렬에 각 커널의 최댓값이 기록된 형태일 것입니다.

이 경우, (2, 2) 가 (1, 1)로 바뀌게 다므로 입력 데이터의 75%를 절약하게 됩니다.

### 7. What is Dropout?

Dropout은 본직적으로 과적합, oerfitting 방지를 위한 조직화 형식으로 생각할 수 있습니다.

학습 도중에 각 Unit은 연결에 따라서 무작위로 Dropout 되면서 유닛의 특정한 학습 세트에 대한 과도한 상호작용, Too Much Co-Adapting 방지에 도움이 됩니다.
