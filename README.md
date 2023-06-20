# Dataset 
Download and put the following datasets in the ```data``` dir. 
- Amharic News Text Classification Dataset [Here](https://github.com/IsraelAbebe/An-Amharic-News-Text-classification-Dataset) 
- Amharic Sentiment Analysis Dataset [Here](https://github.com/uhh-lt/ASAB/tree/main/data)

# How to run

Run the ```main.py``` file with the following arguments:
- ```--model``` Model Type (```transformer``` or ```lstm```)
- ```--dataset``` Dataset to use (```news``` or ```sentiment```)
- ```--batchsize``` (an integer)
- ```--epoch``` (an integer)
- ```--normalize``` to normalize the inputs (```0``` (False) or ```1``` (True))
- ```--trans``` to transliterate the input (```0``` (False) or ```1``` (True))

## Example 

```
python main.py --model lstm --dataset news
```