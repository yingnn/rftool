# rftool

A convenient wrapper for random forest application.

## install

install package on a virtualenv is recommended.

```
pip install git+https://github.com/yingnn/rftool
```

## usage

data format support:

support txt and xlsx(or xls) format data.

in data, one row represents a sample. the first column is sample id. the second column is sample class if supplied, if you want to predict a sample's class, make sample class no value. if a data item is not available, make it empty. in txt file, columns are separated by tab. below is a data demo:

```
id    class    col0    col1    col2    col3

0     class_0    .1      .3      .4    male

1     class_0   .12     .34      .5    female

2     class_1   .09      .6     .45    male

3     class_2           .25     .68    female

4               .11     .54     .42    female
```

in the demo data above, sample with id 4 will be predicted for it's class. sample 3 at column col0 has no value.

run script:

```
rf.py /path/to/a_data_file
```
