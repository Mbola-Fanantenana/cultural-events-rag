{
  "questions_simples": [
    {
      "question": "Quels sont les événements à Châtellerault ?",
      "expected_keywords": ["Châtellerault", "événement"],
      "expected_nb_events": 5
    },
    {
      "question": "Y a-t-il des visites guidées ?",
      "expected_keywords": ["visite", "guidée"],
      "expected_nb_events": 5
    },
    {
      "question": "Quels événements sont gratuits ?",
      "expected_keywords": ["gratuit", "libre"],
      "expected_nb_events": 5
    }
  ],
  
  "questions_specifiques": [
    {
      "question": "Parle-moi du Carillon Bollée",
      "expected_keywords": ["Carillon", "Bollée"],
      "expected_event": "Visite du Carillon Bollée"
    },
    {
      "question": "Quels sont les horaires de la Manu ?",
      "expected_keywords": ["Manu", "horaire"],
      "expected_event": "Visite de la Manu"
    }
  ],
  
  "questions_temporelles": [
    {
      "question": "Quels événements en avril 2026 ?",
      "expected_keywords": ["avril", "2026"],
      "expected_nb_events": 5
    },
    {
      "question": "Que faire ce week-end ?",
      "expected_keywords": ["week-end", "samedi", "dimanche"],
      "expected_nb_events": 5
    }
  ],
  
  "questions_negatives": [
    {
      "question": "Y a-t-il des concerts de rock ?",
      "expected_answer_type": "negative_or_limited",
      "note": "Peu probable d'avoir des concerts de rock dans les données"
    },
    {
      "question": "Quels événements à Paris ?",
      "expected_answer_type": "limited",
      "note": "Peu d'événements à Paris (10%)"
    }
  ]
}
