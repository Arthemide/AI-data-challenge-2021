#feature_cols = []
stade_cols = [
    'Zone anterieure droite bas',
    'Zone anterieure droite haut',
    'Zone anterieure gauche bas',
    'Zone anterieure gauche haut',
    'Zone posterieure droite bas',
    'Zone posterieure droite haut',
    'Zone posterieure gauche bas',
    'Zone posterieure gauche haut',
]

boolean_cols = [
    'Tabagisme actif',
    'BPCO',
    'Asthme',
    'Autre antecedent respiratoire',
    'Hypertension arterielle',
    'Cardiopathie ischemique',
    'Cardiopathie rythmique',
    'Diabete de type 1',
    'Diabete de type 2',
    'Diabetes',
    'Cancer  hemopathie maligne',
    'Demence',
    'Statut immunodeprime',
    "AINS au long cours (dans le cadre d'une pathologie suivie)",
    'AINS ponctuel recent (cadre des symptomatologies COVID-19 suspect avere)',
    'Confusion',
]

int_cols = [
    'Tension arterielle systolique (mmHg)',
    'Tension arterielle diastolique (mmHg)',
    'Frequence cardiaque (puls. min)',
    'Frequence respiratoire (resp. min)',
    'Temperature (Celsius)',
    'Saturation O2',
]

unique_cols = [
    # 'Nom du centre',
    # "id",
    'age',
    'Sexe',
    'Lieu de provenance du patient',
    'Echographiste',
    'Date de debut de la symptomatologie',
    'Oxygenotherapie'
]

feature_cols = stade_cols + boolean_cols + unique_cols + int_cols

def dt64_to_float(dt64):
    if type(dt64) is float and np.isnan(dt64):
        return .0 # np.nan
    try:
        time = datetime.strptime(dt64, '%Y-%m-%d')
        return time.timestamp()
    except Exception as e:
        print(dt64)
        print(e)
    return dt64

def convert(x):
    if x == 'Oui':
        return 1
    return 0

def pre_processing(df):
    dataFrame = df.copy()
    # Change date from object to float
    dataFrame['Date de debut de la symptomatologie'] = dataFrame['Date de debut de la symptomatologie'].map(dt64_to_float).to_numpy()

    # Change option to int
    dataFrame['Oxygenotherapie'] = dataFrame['Oxygenotherapie'].map({"Air ambiant": 1, "Moderee": 2, "Assistance respiratoire": 3, np.nan:0}).to_numpy()

    # Change string to int
    dataFrame['Sexe'] = dataFrame['Sexe'].map({"Masculin": 0, "Feminin": 1}).to_numpy()

    # Change string to int
    dataFrame['Lieu de provenance du patient'] = dataFrame['Lieu de provenance du patient'].map({"Domicile": 0, "EHPAD": 1, "Hopital": 2, "Autre": 3, np.nan:4}).to_numpy()

    # Change option to int
    dataFrame['Echographiste'] = dataFrame['Echographiste'].map({"Forme pour l'epidemie": 1, "Experience d'echographie": 2, "Expert": 3, np.nan:0}).to_numpy()


    dataFrame[boolean_cols] = dataFrame[boolean_cols].apply(lambda value: value.map(lambda x: 1 if x == 'Oui' else 0))

    # Change data from string "Stade X" to int X
    dataFrame[stade_cols] = dataFrame[stade_cols].apply(lambda value: value.map(lambda x: int(x[-1]) if type(x) is str else x))

    return dataFrame
