import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

'''Index(['Employment_period_', 'Time_in_current_department_', 'Gender_',
       'Team_leader_', 'Age_', 'Member_of_professional_organizations_',
       '.Net_', 'SQL_Server_', 'HTML_CSS_Java_Script_', 'PHP_mySQL_',
       'Fast_working', 'Awards', 'Communicative_'],
      dtype='object')'''

data = pd.read_csv("C:/Users/Sony/repo_codes/Mtech-repo/DM-assignment1/Employee_skills_traits.csv")

data.columns = data.columns.str.replace(' ', '_')

## Team lead pattern or criteria
team_lead = data[data['Team_leader_'] == 1].set_index('ID')


# team_lead_.net = data[data['Team_leader_']==1].groupby(['.Net_', 'SQL_Server_'])['Awards'].sum().unstack().reset_index().fillna(0)

# team_lead_php =   data[data['Team_leader_']==1].groupby(['PHP_mySQL_'])['Awards'].sum().unstack().reset_index().fillna(0)
def age_grp(age):
    if age<=26:
        return 'grp26'
    elif age>26 and age<=30:
        return 'grp30'
    elif age>30 and age<=40:
        return 'grp40'
    else:
        return 'grp50'
##data.groupby('Employment_period_').mean()
def exp_level(Employment_period_):
    if Employment_period_<=5:
        return 'epr5'
    elif Employment_period_>5 and Employment_period_<=10:
        return 'epr10'
    elif Employment_period_>10 and Employment_period_<=15:
        return 'epr15'
    else:
        return 'epr20'

##data.groupby('Time_in_current_department_').mean()
def dep_tenure(Time_in_current_department_):
    if Time_in_current_department_<=3:
        return 'dtr2'
    elif Time_in_current_department_>3 and Time_in_current_department_<=6:
        return 'dtr5'
    elif Time_in_current_department_>6 and Time_in_current_department_<=9:
        return 'dtr8'
    else:
        return 'dtr9+'

def hot_encode(x):
    if (x <= 0):
        return 0
    if (x >= 1):
        return 1


team_lead['grp40'] = team_lead['Age_'].transform(age_grp).eq('grp40')
team_lead['grp50'] = team_lead['Age_'].transform(age_grp).eq('grp50')
team_lead['grp26'] = team_lead['Age_'].transform(age_grp).eq('grp26')
team_lead['grp30'] = team_lead['Age_'].transform(age_grp).eq('grp30')

team_lead['epr5'] = team_lead['Employment_period_'].transform(exp_level).eq('epr5')
team_lead['epr10'] = team_lead['Employment_period_'].transform(exp_level).eq('epr10')
team_lead['epr15'] = team_lead['Employment_period_'].transform(exp_level).eq('epr15')
team_lead['epr20'] = team_lead['Employment_period_'].transform(exp_level).eq('epr20')

team_lead['dtr2'] = team_lead['Time_in_current_department_'].transform(dep_tenure).eq('dtr2')
team_lead['dtr5'] = team_lead['Time_in_current_department_'].transform(dep_tenure).eq('dtr5')
team_lead['dtr8'] = team_lead['Time_in_current_department_'].transform(dep_tenure).eq('dtr8')
team_lead['dtr9+'] = team_lead['Time_in_current_department_'].transform(dep_tenure).eq('dtr9+')

tl=team_lead.drop(['Employment_period_','Time_in_current_department_','Age_'],axis='columns')
basket_encoded = team_lead.applymap(hot_encode)

tech_skill = apriori(basket_encoded, min_support=0.25, use_colnames=True)

rules = association_rules(tech_skill, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
pd.set_option('display.max_columns', None)
print(rules)
