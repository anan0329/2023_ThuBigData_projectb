# 2023 IMBD competition project B

Please see [imbd2023](https://imbd2023.thu.edu.tw/) for more information.

## setup

Install `pipenv`, then run `pipenv sync` to install all packages specified in Pipfile.lock

## project structure

```{bash}
tree -L 2

# ├─data
# │  ├─B_traing1 # pre-training data
# │  │  ├─not reworkable
# │  │  └─reworkable
# │  └─B_traing2 # training data
# │      ├─not reworkable
# │      └─reworkable
# ├─FTP # pretrained model dir
# ├─model # finetuned model dir
# └─src
```

## reminder

The source of the data is provided by the competition's official orgaizers. This project only open-sources the related code.
