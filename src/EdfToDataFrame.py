class EdfToDataFrame:
    """Loads an .edf-file, determines seizure onset and saves all markers to a pandas DataFrame"""

    def __init__(self, edf) -> pd.DataFrame:
        self.edf = edf

    def _return_raw(self):
        return mne.io.read_raw(self.edf, preload = True)
    
    def _set_beginning(self, df):
        e_beginning = df[['e-beginn' in x for x in df['description'].str.lower()]]
        s_beginning = df[['s-beginn' in x for x in df['description'].str.lower()]]
        the_beginning = pd.concat([e_beginning, s_beginning], axis=0)
        if the_beginning.empty:
            print(f"Error: No marker containing \"Beginn\" found, cannot determine seizure onset for: {df}")
            print("Setting seizure onset to the beginning of the file")
            onset = "No seizure onset was marked"
            df.loc[-1] = [0, "_Beginn_(assumed)_"]
            df.index = df.index + 1
            df = df.sort_index()
            the_beginning.loc[1,:] = [0, "_Beginn-(assumed)_"]  
        samp_beginn = the_beginning.iloc[0,0].astype(float)
        onset = samp_beginn.astype(float)
        time_from_onset = df["onset"]
        time_from_onset = time_from_onset  - samp_beginn
        df["time_from_onset"] = time_from_onset
        return (df.drop(["onset"], axis = 1), onset)
    
    def _add_source_column(self, df):
        # Add source column to the left
        df["source"] = self.edf.split(folder_splitter)[-1].split(".edf")[0]
        cols = list(df)
        cols.insert(0, cols.pop(cols.index('source')))
        return df.loc[:, cols], df["source"][0]
    
    def _read_config_file(self, config=CONFIG_FILE):
        mEEG = pd.read_excel(CONFIG_FILE, sheet_name="EEG")
        mEEG = mEEG[["mName", "mTranslation", "mSubstitution"]]
        mEEG.dropna(how="all", inplace=True)
        mEEG = mEEG.set_index("mName")

        mSemio = pd.read_excel(CONFIG_FILE, sheet_name="Semio")
        mSemio = mSemio[["mName", "mTranslation", "mSubstitution"]]
        mSemio.dropna(how="all", inplace=True)
        mSemio = mSemio.set_index("mName")

        mModifiers = pd.read_excel(CONFIG_FILE, sheet_name="Modifiers")
        mModifiers = mModifiers[["mName", "mTranslation", "mSubstitution"]]
        mModifiers.dropna(how="all", inplace=True)
        mModifiers = mModifiers.set_index("mName")

        mAnatomy = pd.read_excel(CONFIG_FILE, sheet_name="Anatomy")
        mAnatomy = mAnatomy[["mName", "mTranslation", "mSubstitution"]]
        mAnatomy.dropna(how="all", inplace=True)
        mAnatomy = mAnatomy.set_index("mName")

        return(mEEG, mSemio, mModifiers, mAnatomy)

    def _marker_to_text(self, string, substitute=True):
        """
        Splits the input string as needed
        Translates according to CONFIG_FILE
        returns:
          a string in human readable format
          type: EEG, Semio, Testing
          markers_code: e-"IAmTheBaseName"
        """
        mEEG, mSemio, mModifiers, mAnatomy = self._read_config_file()
        d = dict()
        readbable = str()
        # ignore the i- markers - not need to translate those
        if string.startswith("i-"):
            return "ignored"
        # the rest belongs to one of three groups
        elif string.startswith("e-"):
            d["type"] = "EEG"
        elif string.startswith("s-"):
            d["type"] = "Semiology"
        else:
            d["type"] = "Testing"
    
        # this returns a list of markers and modifiers
        rex = re.findall(r"[-|+]\w*", string)

        # First job is to define the base 
        try:
            # base comes first
            r = rex[0].strip("-")
            rr = rex[0]
            if r in mEEG.index:
                base = mEEG.loc[str(r)][0]
            else:
                base = str(r)
            # now we can drop it from the list
            rex.remove(rr)

        # This might not be a smart move :-(
        except Exception as e:
            print(f"Could not determine base: {e}, setting it to {string}")
            base = string
    
    
        # 2nd job: substitutions
        if substitute == True:
            for r in rex:
                r = r.split("-")[-1].split("+")[-1] 
                if r in mEEG.index:
                    if mEEG.loc[str(r)][1] != None:
                        newitems = list()
                        try:
                            print(f"mEEG.loc[str(r)][1] --> {mEEG.loc[str(r)][1]}")
                            # split the substitution
                            subst = str(mEEG.loc[str(r)][1]).split("-")
                            for s in subst:
                                if not s in rex:
                                    newitems.append(s)
                            for n in newitems:
                                rex.append(str("-" + n))    
                            # delete r, as it has just been substituted
                            rex.remove(str("-" + r))
                        except Exception as e:
                            print(e)
                if r in mSemio.index:
                    pass
                if r in mModifiers.index:
                    pass
                if r in mAnatomy.index:
                    pass
        print(f"rex after substitution   -->   {rex}")      

     #   define placeholder lists
        strEEG = []
        strSemio = []
        strAna = []
        strMod = []
        strNotRecognized = []
    
        # now we can go throug the modifiers etc.
        for r in rex:
            r = r.split("-")[-1] 
            r = r.split("+")[-1]
            r = r.strip("-")     
            if r in mEEG.index:
                strEEG.append(mEEG.loc[str(r)][0])
            elif r in mSemio.index:
                strSemio.append(mSemio.loc[str(r)][0])
            elif str("+" + r) in mModifiers.index:
                strMod.append(str(mModifiers.loc[str("+" + r)][0]))
            elif str(r) in mModifiers.index:
                strMod.append(str("with " + mModifiers.loc[str(r)][0]))
            elif r in mAnatomy.index:
                strAna.append(mAnatomy.loc[str(r)][0])
            else:
                strNotRecognized.append(r)

        # make sure output order is always the same + return 
        readable = ""
        if strEEG is not []:
            #strEEG = set(strEEG)
            for e in sorted(strEEG):
                readable += str(" " + e)
        if strSemio is not []:
            for m in sorted(strSemio):
                readable += str(" " + m)
        if strMod is not []:
            for m in sorted(strMod):
                readable += str(" " + m)
        if strAna is not []:
            for a in sorted(strAna):
                readable += str(" " + a)     
        if strNotRecognized is not []:
            for m in sorted(strNotRecognized):
                readable += str(" " + m)

        # bring back the prefix
        if string.startswith("e-"):
            prefix = "e-"
        elif string.startswith("s-"):
            prefix = "s-"
        else:
            prefix = ""

        readable = prefix + base + " " + readable
        if readable.startswith(" "):
            readable.lstrip(" ")
        return readable
    
    def raw_to_df(self):
        raw = self._return_raw()
        df = pd.DataFrame(raw.annotations)
        df = df.drop(["duration"], axis=1)
        df = df.drop(["orig_time"], axis=1)
        df, onset = self._set_beginning(df)
        df, source = self._add_source_column(df)
        return df, onset
 
    def translate_markers(self, df):
        """Takes a DataFrame as produced by raw_to_df and 
           changes Markers in Column description to human readable form.

        Args:
            df ([pandas.DataFrame]): Output of EdfToDataFrame.raw_to_df
        """
        pass