Logistic regression:
h(X) = g (theta' X)
where g(z) = 1 / (1 + e^(-z))

Training set:
(x1, y1), (x2, y2), (x3, y3)..., (xm, ym)
m examples, x (= Rn+1	x0 = 1, y (= {0,1}

h(theta' X) = 1 / (1 + e^-(theta' X))

Cost function:
-log(h(theta' X)) if y = 1
-log(1 - h(theta' X)) if y = 0
i.e.
-y log(h(theta' X)) - (1 - y)log(1 - h(theta' X))

Hypothesis:
y = 1 if h(x) >= 0.5
theta' X >= 0
y = 0 if h(x) < 0.5
theta' X < 0

