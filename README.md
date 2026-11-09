# TFM - Metaheuristic-Based Instance Selection for Compact Training of Deep Neural Networks

## Resumen

**PALABRAS CLAVE**: Aprendizaje Profundo, Clasificación de Imágenes, Reducción de Datos de Entrenamiento, Selección de Instancias, Transfer Learning, Metaheurísticas, Computación Evolutiva.

El entrenamiento de clasificadores de imágenes basados en aprendizaje profundo sobre grandes conjuntos de datos puede resultar computacionalmente costoso, aunque muchas de las instancias disponibles pueden ser redundantes o aportar poca información adicional.

Este Trabajo de Fin de Máster estudia el problema de selección de instancias con cardinalidad fija bajo un protocolo de evaluación lineal utilizando características congeladas. Se comparan Random Search (RS), Large Neighborhood Search (LNS), un Algoritmo Genético (GA) y un Algoritmo Memético (MA), empleando representaciones preentrenadas en ImageNet y manteniendo congelados los extractores de características.

El estudio sigue un diseño experimental en dos etapas. En una primera fase sobre MNIST se comparan cuatro representaciones profundas —AlexNet, ResNeXt-50, EfficientNetV2-S y Swin-T— junto con los cuatro métodos de búsqueda, utilizando cinco niveles de retención de instancias y cinco semillas. AlexNet junto con GA obtiene el mejor rendimiento agregado y se selecciona de forma descriptiva para la fase principal.

En la segunda etapa, realizada sobre CIFAR-10 y Tiny ImageNet, GA y RS se evalúan con un presupuesto ampliado de 1000 evaluaciones y se comparan con una muestra aleatoria estratificada (SSRS), global k-center greedy (KCG) y el entrenamiento utilizando el 100 % del conjunto base.

Los resultados muestran que es posible reducir sustancialmente la cantidad de datos manteniendo un rendimiento competitivo. En CIFAR-10, GA alcanza una precisión del 80.87 % utilizando el 90 % de los datos, solo 0.08 puntos porcentuales por debajo del entrenamiento con el 100 %. En Tiny ImageNet, KCG alcanza un 45.72 % utilizando únicamente el 25 % de los datos, 0.23 puntos por debajo del conjunto completo. Sin embargo, ningún método de selección domina de forma consistente en todos los escenarios y las estrategias wrapper más costosas ofrecen mejoras pequeñas e inconsistentes frente a alternativas más sencillas. :chatgpt-content-reference{index="1"}

## Abstract

**KEYWORDS**: Deep Learning, Image Classification, Training Data Reduction, Instance Selection, Transfer Learning, Metaheuristics, Evolutionary Computation.

Training image classifiers based on deep learning on large datasets can be computationally demanding, although many examples may be redundant or only marginally informative.

This Master's Thesis studies fixed-cardinality instance selection under a linear evaluation protocol with frozen features. Random Search (RS), Large Neighborhood Search (LNS), a Genetic Algorithm (GA), and a Memetic Algorithm (MA) are compared using cached representations pretrained on ImageNet, with the deep feature extractors kept frozen and only a newly initialized linear classifier trained for each candidate subset.

A two-stage experimental design is adopted. The preliminary MNIST stage compares four frozen representations —AlexNet, ResNeXt-50, EfficientNetV2-S, and Swin-T— over five instance retention levels and five seeds. AlexNet with GA achieves the highest aggregate validation accuracy and is selected descriptively for the main stage.

On CIFAR-10 and Tiny ImageNet, GA and RS are evaluated with an enlarged budget of 1000 candidate evaluations and compared with a Single Stratified Random Sample (SSRS), global k-center greedy (KCG), and the 100% base training baseline.

The results show that substantial data reduction can preserve competitive performance under linear evaluation. On CIFAR-10, GA reaches 80.87% accuracy at 90% retention, only 0.08 percentage points below the 100% baseline. On Tiny ImageNet, KCG reaches 45.72% accuracy at only 25% retention, 0.23 points below the complete base training set. However, no subset selection method consistently dominates across datasets and retention levels, while expensive wrapper search provides only small and inconsistent gains over simpler baselines. :chatgpt-content-reference{index="2"}
