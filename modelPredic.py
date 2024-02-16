from sklearn.neighbors import NearestNeighbors # Voisin proche (?)
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import streamlit as st
from st_clickable_images import clickable_images
[theme]
base="dark"
#Streamlit plein ecran
st.set_page_config(layout="wide")
#url de base pour afficher les images
basImage = "http://image.tmdb.org/t/p/original/"


# LES DATAFRAMES
# dftmdbKnnTrue = pd.read_csv("tableReco\\dftmdbKnn.csv")
dftitlePrincipalActDir = pd.read_csv("dftitlePrincipalActDir.csv")
dftitleAkaFrOutSeri = pd.read_csv('dftitleAkaFrOutSeri.csv')
dftmdbposter = pd.read_csv("dftmdbPoster.csv")
dfCompany = pd.read_csv("companyFilm.csv")
dftmdbKnn = pd.read_csv("dftmdbKnn.csv")
dfPaysCie = pd.read_csv('dfPaysCie.csv')

dftmdbKnn.drop_duplicates(subset='tconst', inplace = True)
# dftitleAkaFrOutSeri.drop_duplicates(subset='titleId', inplace = True)

dftmdbKnnGenre = dftmdbKnn.drop(columns=['production_companies_country', 'budget', 'popularity', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count','production_companies_name']).copy()

# Les session_state
if 'tconst' not in st.session_state:
    st.session_state['tconst']=""


# LES FONCTIONS 

# Recuperation des identifiants des reals, acteurs et compagnie de production
def casting(tconst):
    castMovie = dftitlePrincipalActDir[dftitlePrincipalActDir['tconst']==tconst]
    castDir = castMovie[castMovie['category']=='director']['nconst']
    castAct = castMovie[castMovie['category'].isin(['actress', 'actor'])]['nconst']
    companyFilm = dfCompany[dfCompany["titleId"] == tconst]['production_companies_name']
    companyPaysFilm = dfPaysCie[dfPaysCie["titleId"] == tconst]['production_companies_country']
    listDir = []
    listAct = []
    listCie = []
    listPaysCie = []


    for dir in range(len(castDir)):
        listDir.append(castDir.values[dir])

    for act in range(len(castAct)):
        listAct.append(castAct.values[act])    

    for Cie in range(len(companyFilm)):
        listCie.append(companyFilm.values[Cie])

    if not companyPaysFilm.empty:
        for paysCie in range(len(companyPaysFilm)):
            if pd.notnull(companyPaysFilm.values[paysCie]):
                listPaysCie.append(companyPaysFilm.values[paysCie])

    return listDir, listAct, listCie, listPaysCie

# Fonction moisie pour recuperer des info avec un apply sur une colonne de df
def recup(x, listCie):
    for Cie in listCie:
        if Cie in x :
            return x

# Recuperation des autres films du meme real ou acteur et augmentation de leurs poid dans la df de prediction
def Filmographie(listDir, listAct, listCie, listPaysCie, dfBasePred):
    # Partie Realisateur et acteur
    dfCopy = dfBasePred.copy()
    ListMovDir = dftitlePrincipalActDir[dftitlePrincipalActDir['nconst'].isin(listDir)]['tconst']
    ListMovAct = dftitlePrincipalActDir[dftitlePrincipalActDir['nconst'].isin(listAct)]['tconst']
    colGenre = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
                'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western', 'TVMovie']
    dfCopy.loc[dfCopy['tconst'].isin(ListMovDir), colGenre] *= 3;    
    dfCopy.loc[dfCopy['tconst'].isin(ListMovAct), colGenre] *= 2;    

    # partie compagnie

    # Nom Compagnie
    dfnomCie = dftmdbKnn.copy()
    dfnomCie.dropna(inplace= True)
    dfnomCie['production_companies_name'] = dfnomCie['production_companies_name'].apply(recup, args= (listCie,))
    dfnomCie.dropna(inplace = True)
    listFilmCie = list(dfnomCie['tconst'])
    dfCopy.loc[dfCopy['tconst'].isin(listFilmCie), colGenre] *= 1.5
    # Pays Compagnie
    if listPaysCie:
        dfpaysCie = dftmdbKnn.copy()
        dfpaysCie.dropna(inplace= True)
        dfpaysCie['production_companies_country'] = dfpaysCie['production_companies_country'].apply(recup, args= (listPaysCie,))
        dfpaysCie.dropna(inplace = True)
        listPaysCie = list(dfnomCie['tconst'])
        dfCopy.loc[dfCopy['tconst'].isin(listPaysCie), colGenre] *= 1.5


    return dfCopy


# Tri de la liste de film a partir des films selectionnés par Nearest_Neighbors
def recupNfilm(listFilm, n):
    listReco = []
    listReco.extend(listFilm['tconst'].head(5))
    listReco.extend(listFilm.sort_values('vote_average', ascending=False).head(n)['tconst'].tolist())
    listReco.extend(listFilm.sort_values('revenue', ascending= False).head(n)['tconst'].tolist())
    # listReco.extend(listFilm.sort_values('popularity', ascending= False).head(n)['tconst'].tolist())

    return listReco


# Recuperation des noms des films
def recupTitreFilm(listFilm):
    listTitre = {}
    for i in range(len(listFilm)):
        titre = dftitleAkaFrOutSeri[dftitleAkaFrOutSeri['titleId']==listFilm[i]]['title'].iloc[0]
        if (not listTitre or titre not in listTitre.values()) and titre != nomFilm:
            listTitre[listFilm[i]] = titre
        # Trier le dictionnaire par ordre alphabétique des valeurs
    listTitre_triee = {cle: valeur for cle, valeur in sorted(listTitre.items(), key=lambda item: item[1])}
    return listTitre



# Fonction de prediction
def predict(tconst):
    film = dftmdbKnnGenre[dftmdbKnnGenre['tconst'] == tconst]
    # Tri sur l'animation et family (un film d'animation ne renverra que de l'animation et inversement)
    # if (film['Animation'].iloc[0] == 1) & (film['Family'].iloc[0] == 0):
    #     dfPredict = dftmdbKnnGenre[dftmdbKnnGenre['Animation']==1]

    # if ((film['Animation'].iloc[0] == 0) & (film['Family'].iloc[0] == 0)) & (film['Horror'].iloc[0] == 0):
    #     dfPredict = dftmdbKnnGenre[(dftmdbKnnGenre['Animation']==0) & (dftmdbKnnGenre['Horror']==0)]
    # if film['Horror'].iloc[0] == 0:
    # # ((film['Animation'].iloc[0] == 0) & (film['Family'].iloc[0] == 0)) & (film['Horror'].iloc[0] == 0):
    #     dfPredict = dftmdbKnnGenre[dftmdbKnnGenre['Horror']==0]

    # elif (film['Family'].iloc[0] == 1) & (film['Animation'].iloc[0] == 0):
    #     dfPredict = dftmdbKnnGenre[dftmdbKnnGenre['Family']==1]

    # elif (film['Animation'].iloc[0] == 1) & (film['Family'].iloc[0] == 1):
    #     dfPredict = dftmdbKnnGenre.loc[((dftmdbKnnGenre['Family'] == 1) & (dftmdbKnnGenre['Animation'] == 1))]

    # elif film['Horror'].iloc[0] == 1:
    #     dfPredict = dftmdbKnnGenre[dftmdbKnnGenre['Horror']==1]

    # else:
    dfPredict = dftmdbKnnGenre.copy()


    # On recupere le Casting
    dir, act , Cie, paysCie = casting(tconst)

    # On recupere et on pondere les differents films liées aux différents membres du casting
    dfPredictFinal = Filmographie(dir, act, Cie, paysCie, dfPredict)
    dfPredictFinal.reset_index(inplace = True, drop = True)

    indexFilm = dfPredictFinal[dfPredictFinal['tconst'] == tconst].index[0]


    ########### Modele de Prediction ##########
    Xpredict = dfPredictFinal.drop(columns='tconst').copy()
    modelFilm = NearestNeighbors(n_neighbors=3, metric ='manhattan')
    # On prepare et entraine le modele
    modelFilm.fit(Xpredict)

    # recherche des plus proche voisin : 
    recherche = Xpredict.iloc[indexFilm]

    # On recupere l'indice des films voisins
    distances, indices = modelFilm.kneighbors([recherche], n_neighbors= 50)
    # On recupere toutes les infos des filmVoisins
    filmVoisinGenre = dfPredictFinal.iloc[indices[0]]
    filmVoisinComplet = pd.merge(filmVoisinGenre['tconst'], dftmdbKnn, on = 'tconst', how = 'inner')
    # Evaluation du model

    # Recuperation des films conseillés par differentes méthodes
    listTconst = recupNfilm(filmVoisinComplet, 5)
    lisTitre = recupTitreFilm(listTconst)



    

    return lisTitre


# Fonction affichage des film recommandé
def affiche(lisTitreFilm):
    row1 , row2, row3, row4, row5 = st.columns(4), st.columns(4), st.columns(4),st.columns(4), st.columns(4)

    for i, col in zip(range (len(lisTitreFilm)), row1 + row2 + row3 + row4 + row5 ):
        with col.container():
            cle = list(lisTitreFilm.keys())[i]
            # Utilisez directement la clé de lisTitreFilm au lieu de listTconstFilm[i]
            st.subheader(lisTitreFilm[cle])
            finUrl = dftmdbposter[dftmdbposter['titleId'] == cle]["poster_path"].iloc[0]
            if pd.isna(finUrl) :
                st.image('https://i.ibb.co/2568XWc/notFound.png', use_column_width=True)
            else:
                urlPoster = basImage + finUrl
                st.image(urlPoster, use_column_width=True)





###################################################################################################################
################################# CONTENU DU STREAMLIT ############################################################
###################################################################################################################
st.title("SYSTEME DE RECOMMANDATION DE FILM")
st.header('parfait, infaillible et fantastique !')
st.header('(Ou presque)')
col1, col2 = st.columns(2)
# with col1:
#     st.image("https://i.ibb.co/RzPcD8s/yeahbb.jpg", width=600 )
# with col2:
#     st.image("https://i.ibb.co/4FBNKdL/yeahblack.jpg", width=600)
tconstest = []
# Verification qu'on trouve bien le film dans la table 
with st.sidebar:
    if len(tconstest)==0:
        nomFilm = st.selectbox('Veuillez entrer le film recherché : ', dftitleAkaFrOutSeri['title'], index = None, placeholder="Choose an option")
        tconstest = dftitleAkaFrOutSeri[dftitleAkaFrOutSeri['title'] == nomFilm]['titleId']
    if len(tconstest)==1:
        tconst = dftitleAkaFrOutSeri[dftitleAkaFrOutSeri['title'] == nomFilm]['titleId'].values[0]
        finUrl = dftmdbposter[dftmdbposter['titleId'] == tconst]["poster_path"].iloc[0]
        if pd.isna(finUrl) :
            st.image('https://i.ibb.co/2568XWc/notFound.png', use_column_width=True)
        else:
            urlPoster = basImage + finUrl
            st.image(urlPoster, use_column_width=True)
# Verification que le titre correspond bien à un unique film

if len(tconstest)==1:
    lisTitreFilm = predict(tconst)
    st.subheader(f"Voici des films qui devraient vous interesser : ")
    affiche(lisTitreFilm)


elif len(tconstest) > 1:
    if st.session_state['tconst'] =="":
        st.write(f"il semblerais qu'il y a plusieurs films correspondant à votre recherche dans la base")
        st.write(f"Voici les differents films répondants au nom de {nomFilm}")
        lisTconst = list(tconstest)
        dicoHomo = {}
        for i in range(len(lisTconst)):
            titre = dftitleAkaFrOutSeri[dftitleAkaFrOutSeri['titleId']==lisTconst[i]]['title'].iloc[0]   
            dicoHomo[lisTconst[i]] = titre
        listHomo = []
        # Recuperation et disposition des images de films
        for i in range(len(dicoHomo)):
            cle = list(dicoHomo.keys())[i]
            finUrl = dftmdbposter[dftmdbposter['titleId'] == cle]["poster_path"].iloc[0]
            urlPoster =""
            if pd.isna(finUrl) :
                urlPoster = 'https://i.ibb.co/2568XWc/notFound.png'
            else:
                urlPoster = basImage + dftmdbposter[dftmdbposter['titleId'] == cle]["poster_path"].iloc[0]
            
            listHomo.append(urlPoster)
        # Tentative d'affichage d'image cliquable
        clicked = clickable_images(listHomo,
        titles=[f"Image #{str(i)}" for i in range(4)],    
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "300px"},
        )
        if clicked > -1:
            st.session_state['tconst'] = list(dicoHomo.keys())[clicked]
            st.rerun()


    elif st.session_state['tconst'] != "":
        listFilm = predict(st.session_state['tconst'])
        st.subheader(f"Voici des films qui devraient vous interesser : ")
        affiche(listFilm)
        st.session_state['tconst'] = ""


