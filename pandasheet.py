import pandas as pd
import numpy as np 

pd.set_option('display.max_columns', 0)
# allows us to scroll over the entire pandas dataframe when printing by truncating
pd.options.mode.chained_assignment = None 
# this is going to suppress random errors
df = pd.read_excel('/Users/andrewcrowley/Desktop/JacobsLab/4_7_Olfaction+Sniffing - sub43_1 - Event Logs.xlsx')
expt = pd.read_csv('/Users/andrewcrowley/Desktop/JacobsLab/Andrew/HON_expt_data.csv')
cond3 = pd.read_csv('/Users/andrewcrowley/Desktop/JacobsLab/Andrew/3trials_conditions.csv')
cond4 = pd.read_csv('/Users/andrewcrowley/Desktop/JacobsLab/Andrew/4trials_conditions.csv')
surv = pd.read_excel('/Users/andrewcrowley/Desktop/JacobsLab/Andrew/HON_survey.xlsx')
# imports the Observer output, as well as the experimental data and conditions set for later

df = df.loc[:,'Time_Relative_sf':'Event_Type']
del df["Event_Log"]
# restrict data frame to only necessary columns for clarity

df = df.fillna(method='ffill')

df['alley'] = df['Behavior'].apply(lambda x: 'a' in x)
df['zone'] = df['Behavior'].apply(lambda x: 'z' in x)
df['sniff'] = df['Behavior'].apply(lambda x: 'sniff' in x)
df['hand'] = df['Behavior'].apply(lambda x: 'up' in x or 'raised' in x or 'down' in x)
# add boolean check columns testing whether a behavior is an alley, zone, sniff, or hand raise

df_start = df.loc[df['Event_Type'] == 'State start'] 
df_hand = df.loc[df['hand'] == True]
frames = [df_start, df_hand]
df = pd.concat(frames)
# remove all state stops unless they represent the lowering of a hand

alley_rows = df[(df['alley'] == True)]
alley_rows = alley_rows.loc[:,'Time_Relative_sf':'Event_Type'] # comma necessary because the first colon splits by row, the second by column
alley_rows = alley_rows.rename(columns = {'Behavior':'alley'})
alley_rows = alley_rows[~alley_rows['alley'].apply(lambda x: 'r' in x)] # checks that the column doesn't contain "raised"
alley_rows['alley'] = alley_rows['alley'].map(lambda x: str(x)[1:]) # removes the 'a' now that alleys are in their own column

zone_rows = df[(df['zone'] == True)]
zone_rows = zone_rows.loc[:,'Time_Relative_sf':'Event_Type']
zone_rows = zone_rows.rename(columns = {'Behavior':'zone'})
zone_rows['zone'] = zone_rows['zone'].map(lambda x: str(x)[1:]) # removes the 'z'

sniff_rows = df[(df['sniff'] == True)]
sniff_rows = sniff_rows.loc[:,'Time_Relative_sf':'Event_Type']
sniff_rows = sniff_rows.rename(columns = {'Behavior':'sniff'})

hand_rows = df[(df['hand'] == True)]
hand_rows = hand_rows.loc[:,'Time_Relative_sf':'Event_Type']
hand_rows = hand_rows.rename(columns = {'Behavior':'hand'})
# using the check columns, splits behavior into four different behavior columns

behaviors = [alley_rows, zone_rows, sniff_rows, hand_rows]
goal = pd.concat(behaviors)
# concatenates the behaviors into one data sheet

goal['Subject'] = goal.Observation.str.split('_').str.get(0) # before the underscore
goal['Trial'] = goal.Observation.str.split('_').str.get(1) # after
goal['Subject'] = goal['Subject'].map(lambda x: str(x)[3:]) # removes 'sub'
# split the observation column into a subject and trial number

goal['Zone Duration'] = goal.loc[goal['zone'].notnull(), 'Duration_sf']
goal['Alley Duration'] = goal.loc[goal['alley'].notnull(), 'Duration_sf']
# writes a column that keeps track of how long the subject was present in an alley or zone

goal = goal[['Subject', 'Trial', 'Time_Relative_sf', 'zone', 'alley', 'sniff', 'hand', 'Event_Type', 'Zone Duration', 'Alley Duration', 'Duration_sf']]
goal = goal.sort_values('Time_Relative_sf')
# reorders columns, then sorts the rows by time of occurrence

goal['sniff'] = goal['sniff'].fillna(0)
goal['sniff'].replace('sniff', 1, inplace=True)
goal['hand'].replace('raised', 1, inplace=True)
goal['hand'].replace('up', 1, inplace=True)
goal['hand'].replace('down', 0, inplace=True)
goal.loc[goal['Event_Type'] == 'State stop', 'hand'] = 0
# represent sniffing and raising your hand using 1s and 0s

goal = goal.fillna(method='ffill')
goal['hand'], goal['alley'] = goal['hand'].fillna(0), goal['alley'].fillna('0')
# fill empty cells with applicable data for hand raises and alley position

goal = goal[goal['zone'] != '0']
goal = goal[goal['alley'] != '0']
# removes all 'break' zones and alleys, which occur between zones

goal.drop_duplicates(['Time_Relative_sf'], 'first', inplace=True)
# ensures that there are no conflicting concurrent events

del goal['Event_Type']
goal = goal.rename(columns={'Time_Relative_sf': 'Time', 'Duration_sf': 'Event Duration'})
# rename columns for readability, Event_Type is no longer needed now that the hand column is set up
subj = int(goal.iloc[1,0])
participant = expt[expt['subj_number'] == subj] # experimental row for the correct subject
surv = surv[surv['subj_number'] == subj]

if (subj) < 37:
	cond = cond3
else:
	cond = cond4
order = cond[cond['group'].str.contains(goal.iloc[0,1])] 
# get the testing order for the subject

goal['Guess'] = goal['zone']
correct = {1:'it', 2:'works', 3:'yo'}
goal['Guess'] = goal['Guess'].map(correct)
goal['Correct'] = cond['correct' + goal.iloc[1,1]][1] # make a list with the correct columns, figure out the zones
goal['Guess'] = participant['c' + str(goal.iloc[0,1]) + '_' + str(goal.iloc[5,3])].iloc[0]
# I need to change goal.iloc[0,3] to the contents of zone at the given row!

goal['Condition'] = cond['condition' + goal.iloc[1,1]][1]
goal['Temperature'] = participant['c' + goal.iloc[0,1] + '_temp'].iloc[0]
goal['Nostril_Height'] = participant['Nostril height'].iloc[0]
goal['Nares_Width'] = participant['Nares width'].iloc[0]
goal['Nose_Height'] = participant['Nose height'].iloc[0]
goal['Height'] = participant['Height'].iloc[0]
goal['Age'] = surv['Age'].iloc[0]
goal['Sex'] = surv['Sex'].iloc[0]
goal['Smell_Rate'] = surv['smell_rate'].iloc[0]
# reach into conditions and participant sheets to take out relevant participant statistics

del goal['Event Duration']

goal['Hand Duration'] = np.nan
i = 0
while i < goal.shape[0]:
	if goal['hand'].iloc[i] == 1:
		timeup = goal['Time'].iloc[i]
		up = i
		while goal['hand'].iloc[i]:
			i += 1
		timedown = goal['Time'].iloc[i]
		down = i
		goal['Hand Duration'].iloc[up] = timedown - timeup
		goal['Hand Duration'].iloc[down] = 0
	i += 1
goal = goal.fillna(method='ffill')
goal['Hand Duration'].replace(0, np.nan, inplace=True)
# find the first instance of a hand being raised, find out when the string of 1s/raises stops, then append the elapsed time

goal = goal[['Subject', 'Trial', 'Time', 'zone', 'alley', 'sniff', 'hand', 'Zone Duration', 'Alley Duration', 'Hand Duration', 'Guess', 'Correct', 'Condition', 'Temperature', 'Nostril_Height', 'Nares_Width', 'Height', 'Age', 'Sex', 'Smell_Rate']]
# final column rearrangement to ensure that everything is in order

goal = goal[goal.zone != "Initial zones>"]
goal = goal[goal.alley != "Initial zones>"]
# dropping rows in which zone or alley are blank/state "initial zones", an error that comes about as a result of poor coding

goal.to_csv('goal.csv', index=False)
# export finished csv
