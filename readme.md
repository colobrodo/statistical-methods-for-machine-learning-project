# Kernelized Linear Classification
This is my project for the course Statistical Methods for Machine Learning year: 2023/2024 Kernelized Linear Classification.    
The project is described into `report.pdf`, the Latex source code for it is in the branch `report-source`.     

## Sample usage
I have made two separate commands, the first trains and runs an algorithm with the choosen algorithm and serializes
 it into a pickle object    
```
python .\src\main.py train kernelized-perceptron .models\kernelized-perceptron.pkl
```

The second is used to run a pre-trained model
```
python .\src\main.py -v run .models\kernelized-perceptron.pkl
```