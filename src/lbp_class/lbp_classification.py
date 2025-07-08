import lbp_preproc as pr
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def train_model():
    train_set, train_label = pr.get_dataset(r"C:\Users\dasha\Desktop\bakalarka_data\dith_crop_dataset\train_val_data_mock\train")
    val_set, val_label = pr.get_dataset(r"C:\Users\dasha\Desktop\bakalarka_data\dith_crop_dataset\train_val_data_mock\val")

    # train_set = train_set + val_set
    # train_label = train_label + val_label

    test_set, test_label = pr.get_dataset(r"C:\Users\dasha\Desktop\bakalarka_data\dith_crop_dataset\train_val_data_mock\test")

    le = LabelEncoder()
    le.fit(train_label)
    train_label_int = le.transform(train_label)
    val_label_int = le.transform(val_label)
    test_label_int = le.transform(test_label)


    model = XGBClassifier(n_estimators = 800)
    model.fit(train_set, train_label_int, eval_set=[(train_set, train_label_int), (val_set, val_label_int)])
    prediction = model.predict(test_set)

    acc = accuracy_score(test_label_int, prediction)
    print('Accuracy is', acc)

    model.save_model("model_xgboostClassifier_800est.json")

train_model()
