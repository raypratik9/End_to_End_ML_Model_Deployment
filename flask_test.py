import pandas as pd
from flask import Flask,redirect,url_for,render_template,request,jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle

df=pd.read_csv('preprocessed_data.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
y=df[['loan_status']]
X=df.loc[:,~df.columns.isin(['loan_status'])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
train_column=X_train.columns
sm=SMOTE(random_state=42)
X_train,y_train=sm.fit_resample(X_train,y_train)
scaler = StandardScaler()
scaled=scaler.fit(X_train)



app=Flask(__name__, template_folder='template')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the form data
        print(request.form)
        name = request.form.get('name')
        gender = request.form.get('gender')
        home_status = request.form.get('homeStatus')
        loan_amount = request.form.get('loanAmount')
        emi_duration = request.form.get('emiDuration')
        interest_rate = request.form.get('interestRate')
        employment_length = request.form.get('employmentLength')
        annual_income = request.form.get('annualIncome')

        #processing 
        test_data={}
        home_status=1 if home_status=='Owned' else 0
        employment_length=0 if int(employment_length)<1 else 1
        emi_duration=0 if int(emi_duration)>0 else 1
  
        dti=int(loan_amount)/int(annual_income)
        web_data=[int(loan_amount),emi_duration,float(interest_rate),employment_length,home_status,int(annual_income),1,dti]
        print(web_data)
        data_from_web=[0,1,2,4,5,6,8,9]
        avg_data=[3,7,10,11,12,13,14,15,16,17,18]
        for i in avg_data:
          test_data[X_train.columns[i]]=X_train[X_train.columns[i]].mode()
        for j,i in enumerate(data_from_web):
          test_data[X_train.columns[i]]=web_data[j]
        
        test_data=pd.DataFrame(test_data)
        test_data=test_data[train_column]
        
        test_data=scaled.transform(test_data)
        
        model = pickle.load(open('model.pkl','rb'))
        y_pred =model.predict(test_data)
        
        if y_pred==1:
          return f"Congratulation, you are eligible for loan amount of {loan_amount}"
        else:
          return f"Sorry, you are not eligible for loan amount of {loan_amount} , Try applying for small amount"

        # Return the results along with the original form values
        #return f"{name},Gender={gender}"
        #return render_template('index.html', 
                               #name=name, gender=gender, home_status=home_status,
                               #loan_amount=loan_amount, emi_duration=emi_duration,
                               #interest_rate=interest_rate, employment_length=employment_length,
                               #annual_income=annual_income, loan_status=loan_status)

    return render_template('index.html')

#@app.route('/submit',methods=["GET", "POST"])
#def predict():
#  return jsonify({'loan_status':loan_status})

if __name__=='__main__':
  app.run(host='0.0.0.0',port=5000,debug=True)
