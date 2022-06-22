import numpy as np
import pandas as pd
import scipy
from scipy  import sparse
from sklearn.neighbors import NearestNeighbors
from werkzeug.utils import secure_filename
import re
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rating = pd.read_csv("rating.csv")
places = pd.read_csv("PLacesid.csv")
user_to_place_sparse = sparse.load_npz("sparcematrix.npz")
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_to_place_sparse)
user_to_place = rating.pivot(
    index='user id',
     columns='place name',
      values='rate')

place_list = ['The Egyptian Museum In Cairo', 'The Hanging Church', 'Ben Ezra Synagogue', 'Amr Ibn Al-Aas Mosque', 'Wadi El Natrun monasteries', 'St. Barbara Church', 'Coptic Museum', 'National Museum Of Alexandria', 'Bibliotheca Alexandria', 'Luxor Temple', 'Karnak Temple', 'Valley Of The Kings', 'Valley Of The Queens', 'Deir El Medineh', 'Tel Basta', 'Dahshur Pyramids', 'Giza Plateau', 'Saqqara', 'Royal Jewelry Museum', 'Philae', 'Abu Simbel', 'Mummification Museum', 'Hurghada Museum', 'Sharm al-Sheikh Museum', "Saint Catherine's Monastery", 'Citadel of saladin cairo', 'The Royal Carriages Museum', 'Abdeen Palace', 'Citadel of Qaitbay', 'Museum of Islamic Art', 'National Museum of Egyptian Civilization(NMEC)', 'Baron Empain Palace', 'Cairo Tower', 'Khan el-Khalili', 'Bayt Al-Suhaymi', 'Muhammad Ali Pasha Palace', 'Muhammad Ali Mosque', 'Aswan Museum', 'Temple of Kom Ombo', 'October War Panorama Museum', 'Al-Azhar Mosque', 'Sultan Hassan Mosque', 'Ibn Tulun Mosque', 'Aisha Fahmy Palace', 'The Grand Egyptian Museum', 'Pharaonic Village', 'Sunboat Museum', 'Museum of Islamic Ceramics', 'Egypt Papyrus Museum', 'Bahariya Oasis', 'Agricultural Museum', 'Abu Sir area', 'Mit rahina monuments', 'Greco-Roman Museum', 'Marine Biology Museum', 'Montazah Palace', 'Pompey pillar', 'Temple of Nadura', 'Kom El Shoqafa Cemetery', 'Al-Shatby Tombs', 'black head temple', 'Roman Theater', 'The sunken cities of Abu Qir', 'Cavafy Museum', 'eastern port', 'Fine Arts Museum', 'Mustafa Kamel tombs', 'Deir el-Bahari Temple', 'Mortuary Temple of Amenhotep II', 'Rameseum', 'Temples of Abydos', 'Temple of Horus', 'Temple of Hathor', 'Mortuary Temple of Seti I', 'Khnum temple in Esna', 'Habu city', 'Nubia Museum', 'incomplete obelisk', 'Nile Museum', 'Khnum temple', 'Kalabsha Temple', 'Monastery of Anba Simeon', 'Nubian Village', 'High Dam', 'Mir monuments', 'Koasir el amrana monuments', 'Islamic monuments', 'Coptic monuments', 'Western Mountain Cemetery', 'Temple of Abydos', 'Temple of Ramses II', 'red monastery', 'el-Amarna Hill', 'Virgin Mary Monastery', 'Mallawi Museum', 'Maho Cemetery', 'ancient mosques', 'San  el Hager', 'Tel Ibrahim Awad', 'Ismailia Museum', 'Abu Atwa Tank Museum', 'The International Museum of the Suez Canal Authority', 'Ferdinand de Lesseps house', 'Tree hill', 'sand museum', 'Mini Egypt Park Museum', 'King Tut Museum', 'Ras Muhammed Reserve', 'Nabq Nature Reserve', 'Alf Layla we Layla', 'Monastery of Anba Antonios', 'Monastery of Anba Pola', 'Wadi Hammamet', 'Wadi El-Gemal and Hamata Mountain Reserve', 'Samdayi Reserve (Dolffish House)', 'Moses mountain', 'Serabit traces of servants', 'cave valley', 'al-markh plain', 'Rashid Museum', 'Lighthouse Rashid', 'King Farouk Palace', 'Hibis Temple', 'Dosh temple', 'Bjuwat gabanat', 'Balat Pharaonic Village']
place_dict = {'The Egyptian Museum In Cairo': 0, 'The Hanging Church': 1, 'Ben Ezra Synagogue': 2, 'Amr Ibn Al-Aas Mosque': 3, 'Wadi El Natrun monasteries': 4, 'St. Barbara Church': 5, 'Coptic Museum': 6, 'National Museum Of Alexandria': 7, 'Bibliotheca Alexandria': 8, 'Luxor Temple': 9, 'Karnak Temple': 10, 'Valley Of The Kings': 11, 'Valley Of The Queens': 12, 'Deir El Medineh': 13, 'Tel Basta': 14, 'Dahshur Pyramids': 15, 'Giza Plateau': 16, 'Saqqara': 17, 'Royal Jewelry Museum': 18, 'Philae': 19, 'Abu Simbel': 20, 'Mummification Museum': 21, 'Hurghada Museum': 22, 'Sharm al-Sheikh Museum': 23, "Saint Catherine's Monastery": 24, 'Citadel of saladin cairo': 25, 'The Royal Carriages Museum': 26, 'Abdeen Palace': 27, 'Citadel of Qaitbay': 28, 'Museum of Islamic Art': 29, 'National Museum of Egyptian Civilization(NMEC)': 30, 'Baron Empain Palace': 31, 'Cairo Tower': 32, 'Khan el-Khalili': 33, 'Bayt Al-Suhaymi': 34, 'Muhammad Ali Pasha Palace': 35, 'Muhammad Ali Mosque': 36, 'Aswan Museum': 37, 'Temple of Kom Ombo': 38, 'October War Panorama Museum': 39, 'Al-Azhar Mosque': 40, 'Sultan Hassan Mosque': 41, 'Ibn Tulun Mosque': 42, 'Aisha Fahmy Palace': 43, 'The Grand Egyptian Museum': 44, 'Pharaonic Village': 45, 'Sunboat Museum': 46, 'Museum of Islamic Ceramics': 47, 'Egypt Papyrus Museum': 48, 'Bahariya Oasis': 49, 'Agricultural Museum': 50, 'Abu Sir area': 51, 'Mit rahina monuments': 52, 'Greco-Roman Museum': 53, 'Marine Biology Museum': 54, 'Montazah Palace': 55, 'Pompey pillar': 56, 'Temple of Nadura': 57, 'Kom El Shoqafa Cemetery': 58, 'Al-Shatby Tombs': 59, 'black head temple': 60, 'Roman Theater': 61, 'The sunken cities of Abu Qir': 62, 'Cavafy Museum': 63, 'eastern port': 64, 'Fine Arts Museum': 65, 'Mustafa Kamel tombs': 66, 'Deir el-Bahari Temple': 67, 'Mortuary Temple of Amenhotep II': 68, 'Rameseum': 69, 'Temples of Abydos': 70, 'Temple of Horus': 71, 'Temple of Hathor': 72, 'Mortuary Temple of Seti I': 73, 'Khnum temple in Esna': 74, 'Habu city': 75, 'Nubia Museum': 76, 'incomplete obelisk': 77, 'Nile Museum': 78, 'Khnum temple': 79, 'Kalabsha Temple': 80, 'Monastery of Anba Simeon': 81, 'Nubian Village': 82, 'High Dam': 83, 'Mir monuments': 84, 'Koasir el amrana monuments': 85, 'Islamic monuments': 86, 'Coptic monuments': 87, 'Western Mountain Cemetery': 88, 'Temple of Abydos': 89, 'Temple of Ramses II': 90, 'red monastery': 91, 'el-Amarna Hill': 92, 'Virgin Mary Monastery': 93, 'Mallawi Museum': 94, 'Maho Cemetery': 95, 'ancient mosques': 96, 'San  el Hager': 97, 'Tel Ibrahim Awad': 98, 'Ismailia Museum': 99, 'Abu Atwa Tank Museum': 100, 'The International Museum of the Suez Canal Authority': 101, 'Ferdinand de Lesseps house': 102, 'Tree hill': 103, 'sand museum': 104, 'Mini Egypt Park Museum': 105, 'King Tut Museum': 106, 'Ras Muhammed Reserve': 107, 'Nabq Nature Reserve': 108, 'Alf Layla we Layla': 109, 'Monastery of Anba Antonios': 110, 'Monastery of Anba Pola': 111, 'Wadi Hammamet': 112, 'Wadi El-Gemal and Hamata Mountain Reserve': 113, 'Samdayi Reserve (Dolffish House)': 114, 'Moses mountain': 115, 'Serabit traces of servants': 116, 'cave valley': 117, 'al-markh plain': 118, 'Rashid Museum': 119, 'Lighthouse Rashid': 120, 'King Farouk Palace': 121, 'Hibis Temple': 122, 'Dosh temple': 123, 'Bjuwat gabanat': 124, 'Balat Pharaonic Village': 125}

@app.post('/predict')
def predict(user:int,place:str ,n:int):
    recomened,similar_users = [],[]
    knn_input = np.asarray([user_to_place.values[user - 1]])
    distances, indices = knn_model.kneighbors(knn_input, n_neighbors=n + 1)

    for i in range(1, len(distances[0])):
        similar_users.append(str(str(i)+". User: "+str(indices[0][i] +1)+" separated by distance of "+str(distances[0][i])))
    index = place_dict[place]
    knn_input = np.asarray([user_to_place.values[index]])
    n = min(len(place_list) - 1, n)
    distances, indices = knn_model.kneighbors(knn_input, n_neighbors=n + 1)

    for i in range(1, len(distances[0])):
        recomened.append((place_list[indices[0][i]]))
    return {'Recomened':recomened,'similar_users':similar_users}

result =  predict(5,'The Egyptian Museum In Cairo',3)
print((result))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)