import pandas as pd
import json

if __name__ == '__main__':

    raw_dat = pd.read_csv('trialdata.csv', names=["WID","Index","TimeStamp","Data"])
    u_dat = []
    for wid, dat in zip(raw_dat.WID, raw_dat.Data):
        pdat = json.loads(dat)
        if pdat['phase'] == 'TEST':
            pdat['WID'] = wid
            u_dat.append(pdat)

    parse_dat = pd.DataFrame(u_dat)
    parse_dat.to_csv("parsed_trialdata.csv", index=False)
