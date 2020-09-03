#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

def preprocess_and_analysis():
    ### Import csv file as panda data frame
    df=pd.read_csv('Employee_skills_traits.csv')
    print("\n-------- Input data -----------------\n")
    print(df.head())

    print(df.info())


    ### Drop null values
    df2=df.dropna()
    print("\n-------- after dropping null values -----------------\n")
    print(df2)
    print(df2.columns)
    print(df2.info())
    print(df2.describe())
    print(df2['ID'].value_counts())


    ### Drop duplicate records

    df2.drop_duplicates(subset = 'ID', keep = 'first', inplace = True)
    print("\n-------- Drop duplicate records -----------------\n")
    print(df2)

    print(df2['ID'].value_counts())
    print(df2.info())


    ### Data Analysis - Employment period
    print("\n-------- Data Analysis - Employment period -----------------\n")
    df2.hist(column='Employment period ')
    print(df2['Employment period '].describe())
    plt.ion()
    plt.boxplot(df2['Employment period '])
    plt.pause(0.05)


    ### Data Analysis - Time in current department
    print("\n-------- Data Analysis - Time in current department -----------------\n")
    print(df2.hist(column='Time in current department '))
    print(df2['Time in current department '].describe())
    plt.boxplot(df2['Time in current department '])


    ### Data Analysis - Age
    print("\n-------- Data Analysis - Age -----------------\n")
    df2.hist(column='Age ')
    print(df2['Age '].describe())
    plt.boxplot(df2['Age '])
    plt.pause(0.05)


    ### Data Analysis - Gender
    print("\n-------- Data Analysis - Gender -----------------\n")
    print(df2['Gender '].value_counts())


    ### Data Analysis - Team leader
    print("\n-------- Data Analysis - Team leader -----------------\n")
    print(df2['Team leader '].value_counts())


    ### Data Analysis - Member of professional organizations
    print("\n-------- Data Analysis -  Member of professional organizations -----------------\n")
    print(df2['Member of professional organizations '].value_counts())


    ### Data Analysis - .Net
    print("\n-------- Data Analysis -  .Net -----------------\n")
    print(df2['.Net '].value_counts())


    ### Data Analysis - SQL Server
    print("\n-------- Data Analysis -  SQL Server -----------------\n")
    print(df2['SQL Server '].value_counts())


    ### Data Analysis - HTML CSS Java Script
    print("\n-------- Data Analysis -  HTML CSS Java Script -----------------\n")
    print(df2['HTML CSS Java Script '].value_counts())


    ### Data Analysis - PHP mySQL
    print("\n-------- Data Analysis -  PHP mySQL -----------------\n")
    print(df2['PHP mySQL '].value_counts())


    ### Data Analysis - Fast working
    print("\n-------- Data Analysis -  Fast working -----------------\n")
    print(df2['Fast working'].value_counts())


    ### Data Analysis - Awards
    print("\n-------- Data Analysis -  Awards -----------------\n")
    print(df2['Awards'].value_counts())


    ### Data Analysis - Communicative
    print("\n-------- Data Analysis -  Communicative -----------------\n")
    print(df2['Communicative '].value_counts())


    ### Data Analysis - Age & Employment period
    print("\n-------- Data Analysis -  Age & Employment period -----------------\n")
    fig, ax = plt.subplots()
    ax.scatter(df2['Age '], df2['Employment period '])
    plt.show()
    plt.pause(0.05)

    print(print(df2['Age '].corr(df['Employment period '])))


    ### Data Analysis - Age & Time in current department
    print("\n-------- Data Analysis -  Age & Time in current department -----------------\n")
    fig, ax = plt.subplots()
    ax.scatter(df2['Age '], df2['Time in current department '])
    plt.show()
    plt.pause(0.05)

    print(df2['Age '].corr(df['Time in current department ']))


    ### Data Analysis - Employment period & Time in current department
    print("\n-------- Data Analysis -  Employment period & Time in current department -----------------\n")
    fig, ax = plt.subplots()
    ax.scatter(df2['Employment period '], df2['Time in current department '])
    plt.show()
    plt.pause(0.05)


    print(df2['Employment period '].corr(df['Time in current department ']))


    ### Data Transformation - Binning
    print("\n-------- Data Transformation -----------------\n")
    cut_labels_4 = ['entry', 'mid', 'sr', 'exec']
    cut_bins = [1, 3, 8, 15, 20]
    df2['cut_Employment period'] = pd.cut(df2['Employment period '], bins=cut_bins, labels=cut_labels_4)


    cut_labels_4 = ['fresher', 'senior', 'experienced']
    cut_bins = [1, 2, 8, 12]
    df2['cut_Time in current department'] = pd.cut(df2['Time in current department '], bins=cut_bins, labels=cut_labels_4)


    cut_labels_4 = ['youngster', 'middle_aged', 'old']
    cut_bins = [24, 30, 50, 55]
    df2['cut_Age'] = pd.cut(df2['Age '], bins=cut_bins, labels=cut_labels_4)
    print(df2.head())


    ### Data Transformation (conversion)
    df3 = pd.get_dummies(df2)
    print(df3.head())

    df3.columns

    ip = df3.drop(['ID', 'Employment period ', 'Time in current department ', 'Age '], 1)
    print(ip.head())

    print(ip.describe())

    print(ip.columns)
    print(ip.info())
    print("\n-------- Preprocessinng & Analysis completed -----------------\n")

    return ip


def get_frq_items(ip, min_support):
    return apriori(ip, min_support, use_colnames = True)


def get_association_rules(frq_items):
    rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
    return rules.sort_values(['confidence', 'lift'], ascending =[False, False])


def execute(ip, min_support):
    frq_items = get_frq_items(ip, min_support)
    rules = get_association_rules(frq_items)
    return rules


def main():
    ip = preprocess_and_analysis()

    print("\n-------- Min support = 0.1 --------------------\n")
    print(execute(ip, 0.1).head())
    print("\n-------- Min support = 0.2 -----------------\n")
    print(execute(ip, 0.2).head())
    print("\n-------- Min support = 0.25 -----------------\n")
    print(execute(ip, 0.25).head())
    print("\n-------- Min support = 0.3 -----------------\n")
    print(execute(ip, 0.3).head())
    print("\n-------- Min support = 0.4 -----------------\n")
    print(execute(ip, 0.4))
    print("\n-------- Min support = 0.5 -----------------\n")
    print(execute(ip, 0.5))
    print("\n-------- Min support = 0.6 -----------------\n")
    print(execute(ip, 0.6))

    output = open('output_file.txt', 'w')
    ### We get best results with min support = 0.2
    frq_items = apriori(ip, min_support = 0.2, use_colnames = True)
    output.write("\n--------------Frequent Items & support Count = 0.2---------------------\n")
    output.write(frq_items.to_string(header = True, index = True))
    rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
    output.write("\n\n\n\n--------------Association rules---------------------\n")
    output.write(rules.to_string(header = True, index = True))
    output.close()

main()
