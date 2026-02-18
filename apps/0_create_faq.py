import pandas as pd
import os

# Assicuriamoci che la cartella esista
if not os.path.exists("data"):
    os.makedirs("data")

# Dati delle FAQ (Knowledge Base)
data = {
    "domanda": [
        "Come posso resettare la mia password aziendale?",
        "Qual è la politica per il lavoro da remoto (smart working)?",
        "Come richiedo un giorno di ferie?",
        "Quali sono gli orari della mensa aziendale?",
        "Posso usare il mio laptop personale per lavoro?",
        "Chi devo contattare per problemi con la VPN?",
        "Come funziona il rimborso spese trasferta?",
        "Qual è il massimale per le spese del pranzo durante le trasferte?",
        "Dove trovo i cedolini dello stipendio?",
        "L'azienda offre corsi di formazione?",
        "Qual è la procedura in caso di malattia?",
        "Posso portare ospiti esterni in ufficio?",
        "Come configuro la stampante del secondo piano?",
        "Qual è la password del Wi-Fi ospiti?",
        "Cosa faccio se perdo il badge aziendale?",
    ],
    "risposta": [
        "Per resettare la password, visita il portale https://id.techcorp.com e clicca su 'Password Dimenticata'. Riceverai un codice via SMS.",
        "I dipendenti possono lavorare da remoto fino a 2 giorni a settimana, previa approvazione del manager diretto. I giorni devono essere segnati su WorkDay.",
        "Le ferie vanno richieste tramite il portale HR almeno 3 giorni prima. Il manager ha 24 ore per approvare.",
        "La mensa è aperta dalle 12:00 alle 14:30 dal lunedì al venerdì. Il bar è aperto dalle 08:30 alle 18:00.",
        "No, per motivi di sicurezza è vietato connettere dispositivi personali alla rete interna o usarli per trattare dati aziendali. Usa solo il laptop fornito.",
        "Per problemi VPN, apri un ticket su Jira Service Desk selezionando la categoria 'Network'. In caso di urgenza, chiama l'interno 5555.",
        "Le note spese devono essere caricate su Concur entro il 5 del mese successivo. Allegare sempre le foto degli scontrini.",
        "Il massimale per il pranzo in trasferta è di 25 euro a persona. Cene di rappresentanza richiedono pre-approvazione.",
        "I cedolini sono disponibili nell'area riservata del portale ADP. Vengono caricati il 27 di ogni mese.",
        "Sì, ogni dipendente ha un budget di 500 euro annui per corsi su Udemy o Coursera. Contatta HR per l'attivazione.",
        "In caso di malattia, avvisa il manager entro le 09:00 e invia il numero di protocollo del certificato medico a hr@techcorp.com entro 24 ore.",
        "Gli ospiti devono essere registrati alla reception. Riceveranno un badge temporaneo e devono essere sempre accompagnati da un dipendente.",
        "Per la stampante 2P, cerca l'IP 192.168.1.50 nelle impostazioni stampanti. I driver si installano automaticamente.",
        "La rete Wi-Fi è 'TechCorp_Guest'. La password cambia ogni lunedì ed è visibile sui monitor alla reception.",
        "Denuncia immediatamente lo smarrimento alla sicurezza (sicurezza@techcorp.com) per disattivare il badge. Il costo per il duplicato è di 10 euro.",
    ],
    "categoria": [
        "IT", "HR", "HR", "Servizi", "IT", 
        "IT", "Amministrazione", "Amministrazione", "HR", "HR", 
        "HR", "Sicurezza", "IT", "IT", "Sicurezza"
    ]
}

# Creazione del DataFrame
df = pd.DataFrame(data)

# Salvataggio in CSV
csv_path = "data/faq.csv"
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"File creato con successo: {csv_path}")
print(f"Numero di righe: {len(df)}")
print("\nPrime 3 righe:")
print(df.head(3))