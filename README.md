# Flight Route Optimization using Machine Learning

This project presents a machine learning-based system to predict flight delays and optimize flight routes by analyzing historical flight data. It aims to reduce travel time, cost, and environmental impact by offering optimal flight paths between airports.

## Project Overview

Flight delays and inefficient route planning are major challenges in aviation. This project:
- Predicts flight delays using various classification models.
- Optimizes routes based on both delay probabilities and distances.
- Utilizes graph algorithms (like Dijkstra) for pathfinding.
- Combines predictive models with route optimization to improve air travel efficiency.

## Dataset

The project uses the [`flights.csv` dataset](https://github.com/mdrilwan/datasets/blob/master/flights.csv), which includes:
- Origin and destination airports (with coordinates),
- Distance and travel time,
- Weather data,
- Delay status and type.

These features are used for classification and route optimization tasks.

## Methods

The following machine learning models were implemented:

### Linear Regression

This model predicts delay status using the equation:

```
y = w0 + w1 * x1
```

Where:
- `y`: predicted output (probability of delay),
- `x1`: distance in kilometers,
- `w0`: bias term,
- `w1`: weight for distance.

**Loss Function (MSE):**

```
MSE = (1/n) * Σ(y_i - ŷ_i)^2
```

Where:
- `y_i`: actual label,
- `ŷ_i`: predicted value,
- `n`: number of samples.

### Logistic Regression

The logistic function is used to estimate delay probability:

```
P(y=1 | x) = 1 / (1 + e^-(w0 + w1 * x1))
```

### Decision Tree

- Splits data using **Information Gain**:
  
```
Gain(S, A) = Entropy(S) - Σ(|S_v| / |S|) * Entropy(S_v)
```

Where `S` is the dataset and `A` is the attribute.

### Naïve Bayes

Assumes independence between features. The Gaussian Naïve Bayes formula:

```
P(x_i | y) = (1 / √(2πσ²)) * exp(-(x_i - μ)² / (2σ²))
```

### Support Vector Machine

SVM finds the optimal separating hyperplane:

```
f(x) = w • x + b
```

Decision rule:

```
If f(x) > 0 → class 1, else → class -1
```

**SVM Optimization:**

```
minimize: (1/2) * ||w||^2
subject to: y_i(w • x_i + b) ≥ 1 for all i
```

**SVR (Regression version):**

```
minimize: (1/2) * ||w||² + C * Σ(ξ_i + ξ_i*)
subject to:
    y_i - (w • x_i + b) ≤ ε + ξ_i
    (w • x_i + b) - y_i ≤ ε + ξ_i*
    ξ_i, ξ_i* ≥ 0
```

## Evaluation

Each model was evaluated with:
- Accuracy (Training, Testing, Cross-Validation),
- Mean Squared Error (MSE),
- F1 Score.

| Model             | Train Accuracy (%) | Test Accuracy (%) | CV Accuracy (%) | MSE  | F1 Score |
|------------------|--------------------|--------------------|------------------|------|----------|
| Linear Regression| 76.55              | 80.61              | 77.78            | 0.17 | -        |
| Logistic Regression| 53.12            | 54.08              | 53.40            | 0.24 | 0.58     |
| Naïve Bayes      | 55.97              | 56.12              | 52.85            | 0.25 | 0.58     |
| SVM              | 57.67              | 56.12              | 57.65            | 0.69 | 0.59     |
| Decision Tree    | 90.35              | 92.31              | 89.22            | -    | -        |

## Best Model

- **Linear Regression**: Best accuracy and simple interpretability.
- **Decision Tree**: Effective in classifying delay types (weather, carrier, etc.).

## Installation

```bash
git clone https://github.com/your-username/flight-route-optimization.git
cd flight-route-optimization
```

### Requirements

```bash
pip install requirements.txt
```

## Usage

1. Place the `flights.csv` file in the project root directory.
2. Run the main script:

```bash
python main.py
```

3. *(Optional)* Use scripts in `/visualizations` for plots.

## Future Work

- Incorporate real-time weather and congestion data.
- Extend to international flight networks.
- Apply ensemble or deep learning models.
- Introduce real-time graph updates.

## References

1. Kim, J. (2021). *Data-driven approach using machine learning for real-time flight path optimization*. [Link](https://repository.gatech.edu/entities/publication/7b9a5aa7-e51c-4b67-bde1-78d3e9a4ad0d)
2. Oza, S., et al. (2017). *Flight delay prediction using weighted multiple linear regression*. [Link](https://www.ijecs.in/index.php/ijecs/article/view/1764)
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
4. Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press.
5. Zhang, H. (2004). *The Naive Bayes Classification: A Tutorial*. University of Ottawa.
6. KDnuggets. (2020). *Decision Tree Algorithm Explained*. [Link](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)
7. Analytics Vidhya. (2021). *SVM: A Complete Guide*. [Link](https://www.analyticsvidhya.com/blog/2021/10/support-vector-machinessvm-a-complete-guide-for-beginners/)
8. Dataset: [flights.csv](https://github.com/mdrilwan/datasets/blob/master/flights.csv)

## Team
| Name | URL |
|------|------|
|Gizem EROL| [GitHub](https://github.com/Gizemexe)|
|Selvinaz Zeynep KIYIKCI| [GitHub](https://github.com/selvikiyikci)|
|Zeynep Sude KIRMACI| [GitHub](https://github.com/zeynepkrmc)|


