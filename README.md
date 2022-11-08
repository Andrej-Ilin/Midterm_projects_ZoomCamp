# Mobile Price Classification
## classify mobile price range and description of the problem
About Dataset
Context
Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.

He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.

Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.

In this problem you do not have to predict actual price but a price range indicating how high the price is.

This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).

https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv

### Data preparation and data clearning , EDA, feature importance analysis, Model selection process and parameter tuning.
[05-train-price-model.ipynb](https://github.com/Andrej-Ilin/Midterm_projects_ZoomCamp/blob/main/my_first_project/05-train-price-model.ipynb)

### Training the final model and saving it to a file (e.g. pickle)
[train.py](https://github.com/Andrej-Ilin/Midterm_projects_ZoomCamp/blob/main/my_first_project/train.py)

### Loading the model and serving it via a web serice with Flask
[predict.py](https://github.com/Andrej-Ilin/Midterm_projects_ZoomCamp/blob/main/my_first_project/predict.py)

[Dockerfile](https://github.com/Andrej-Ilin/Midterm_projects_ZoomCamp/blob/main/my_first_project/Dockerfile)  - *If you are using Linux, you should change the line to <ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]> in Dockerfile*



  # *Instuction for run*

- go to the my_first_project folder on the command line

$ docker build -t zoomcamp-test .

$ docker run -it --rm -p 9696:9696 zoomcamp-test

- run your jupyter notebook and run next line in:
__________________________________________
import requests
__________________________________________
url = "http://localhost:9696/predict"
__________________________________________
customer = {
            "battery_power": 1043.0,
            "blue": 1.0,
            "clock_speed": 1.8,
            "dual_sim": 1.0,
            "fc": 14.0,
            "four_g": 0.0,
            "int_memory": 5.0,
            "m_dep": 0.1,
            "mobile_wt": 193.0,
            "n_cores": 3.0,
            "pc": 16.0,
            "px_height": 226.0,
            "px_width": 1412.0,
            "ram": 3476.0,
            "sc_h": 12.0,
            "sc_w": 7.0,
            "talk_time": 2.0,
            "three_g": 0.0,
            "touch_screen": 1.0,
            "wifi": 0.0
           }
___________________________________________
response = requests.post(url, json=customer).json()
response
__________________________________________
