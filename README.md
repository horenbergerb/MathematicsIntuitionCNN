# Modeling Mathematical Intuition With CNNs

## Overview

This project was inspired by the paper [*Intuitive Mathematics: Theoretical and Educational Implications*](https://gseacademic.harvard.edu/~starjo/papers/intuition.pdf). They mention that mathematical intuition can be studied by focusing on cases where intuition fails. They give a particular example (quotation and figure from paper),

```
The error consisted of confusing the y-intercept of a parabola with its vertex (i.e., the “visual” center of the graph).  For example, students who commit this error would decide that the y-intercept in the parabola marked “a” below is “-1,” instead of “-.6.”  Dugdale explains the confusion between the y-intercept and the vertex by pointing out that in previous examples students were given, the y-intercept had always coincided with the vertex of the parabola (see the parabolas c and b below).  The students had thus invented a functional invariance between the two features.
```

![Example from paper](resources/parabola_example.png, "Parabola Example")

I wanted to see if we could use machine learning to model this kind of mistake in forming intuition.

The code in this repository is capable of generating datasets of parabola images and their corresponding y-intercepts as a probability distribution over the image.

It also has a CNN which can be trained on this data and then analyzed.

Right now, the workflow from start to finish would be

```
python3 make_parabolas.py
python3 train.py
```

The performance is decent, but I want to tweak either the model or the dataset to cause a similar mistake to the one described in the cited paper. I successfully achieved this with one iteration of the model, but I was being silly and didn't save the model...

## Example outputs

![Example prediction](resources/example_output.png, "Example prediction from CNN")