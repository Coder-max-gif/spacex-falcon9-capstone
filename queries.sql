-- Unique sites
SELECT DISTINCT Launch_Site FROM launches ORDER BY Launch_Site;

-- First five CCA records
SELECT Date, Booster_Version, Launch_Site FROM launches WHERE Launch_Site LIKE 'CCA%' ORDER BY Date LIMIT 5;

-- NASA CRS payload
SELECT SUM(PAYLOAD_MASS__KG_) AS payload_kg FROM launches WHERE Customer = 'NASA (CRS)';

-- All NASA-related payload
SELECT SUM(PAYLOAD_MASS__KG_) AS payload_kg FROM launches WHERE Customer LIKE '%NASA%';

-- F9 v1.1 average payload
SELECT AVG(PAYLOAD_MASS__KG_) AS average_kg FROM launches WHERE Booster_Version LIKE 'F9 v1.1%';

-- First successful ground landing
SELECT MIN(Date) AS first_date FROM launches WHERE Landing_Outcome='Success (ground pad)';

-- Drone landings with 4000-6000 kg
SELECT Booster_Version, PAYLOAD_MASS__KG_ AS kg FROM launches WHERE Landing_Outcome='Success (drone ship)' AND PAYLOAD_MASS__KG_>4000 AND PAYLOAD_MASS__KG_<6000;

-- Mission outcome counts
SELECT Mission_Outcome, COUNT(*) AS count FROM launches GROUP BY Mission_Outcome;

-- Maximum payload boosters
SELECT Booster_Version, PAYLOAD_MASS__KG_ AS kg FROM launches WHERE PAYLOAD_MASS__KG_=(SELECT MAX(PAYLOAD_MASS__KG_) FROM launches);

-- 2015 drone landing failures
SELECT Date, Booster_Version, Launch_Site FROM launches WHERE Date LIKE '2015%' AND Landing_Outcome='Failure (drone ship)';

-- Landing outcome ranks
SELECT Landing_Outcome, COUNT(*) AS count FROM launches WHERE Date BETWEEN '2010-06-04' AND '2017-03-20' GROUP BY Landing_Outcome ORDER BY count DESC, Landing_Outcome;

-- Site success rates
SELECT Launch_Site, COUNT(*) AS launches, SUM(CASE WHEN Landing_Outcome LIKE 'Success%' THEN 1 ELSE 0 END) AS successful_landings FROM launches GROUP BY Launch_Site;