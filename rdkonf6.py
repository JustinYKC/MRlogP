#!/usr/bin/env python
# //////////////////////////////
# / THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# / EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# / MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# / IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# / OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# / ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# / OTHER DEALINGS IN THE SOFTWARE.

# / You may not distribute copies of the programs source code in any medium.
# / You may not modify the program or any portion of it, thus forming a work
# / based on the program.
# /-------------------------------------------------------------------------
# / Copyright 2018 Steven Shave (stevenshave@gmail.com)
# / rdkonf - v1.04
# / Implementation of high quality RDKit conformer generator as described in:
# / Ebejer, Jean-Paul, Garrett M. Morris, and Charlotte M. Deane.
# / "Freely available conformer generation methods: how good are they?."
# / Journal of chemical information and modeling 52.5 (2012): 1146-1158.
# //////////////////////////////
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import time

class RDKonf():

    def shouldWeKeep(self, keep, mol, q):
        for k in keep:
            if AllChem.GetBestRMS(mol, mol, refId=k[1], prbId=q) <= 0.35:
                # if(AllChem.GetBestRMS(mol, mol, refConfId=k[1], probeConfId=q)<=0.35):   #  Deprecated - previous argument calls to GetBestRMS
                # print "No! - ", AllChem.GetBestRMS(mol, mol, refConfId=k[1], probeConfId=q)
                return False
        return True

    def smiles_to_3dmol(self, smiles:str, title:str=""):
        def shouldWeKeep(keep, mol, q):
            for k in keep:
                if AllChem.GetBestRMS(mol, mol, refId=k[1], prbId=q) <= 0.35:
                    # if(AllChem.GetBestRMS(mol, mol, refConfId=k[1], probeConfId=q)<=0.35):   #  Deprecated - previous argument calls to GetBestRMS
                    # print "No! - ", AllChem.GetBestRMS(mol, mol, refConfId=k[1], probeConfId=q)
                    return False
            return True

        splitsmiles=smiles.split()
        if len(splitsmiles)>1:
            smiles=splitsmiles[0]
            title="".join(splitsmiles[1:])
        mol=Chem.MolFromSmiles(smiles)
        if mol:
            e = []
            keep = []
            n = 300
            mol = Chem.AddHs(mol)
            mol = Chem.RemoveHs(mol)
            nrot = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
            if nrot <= 12:
                n = 200
                if nrot <= 7:
                    n = 50
            confIds = AllChem.EmbedMultipleConfs(mol, n)
            for confId in confIds:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=confId)
                AllChem.UFFOptimizeMolecule(mol, confId=confId)
                e.append(ff.CalcEnergy())
                d = sorted(zip(e, confIds))
            for conf in d:
                if len(keep) < 1:
                    if self.shouldWeKeep(keep, mol, conf[1]):
                        keep.append(conf)
            if len(keep)==1:
                mol.SetProp("_Name", title)
                return mol
        return None

    def rdkonf_use_files(self, input_file: Path):
        output_file = str(input_file) + ".sdf"
        numtokeep = 1
        removehydrogens = 1
        t00 = time.time()
        writer = Chem.SDWriter(str(output_file))
        mol_counter = 0
        smiles_f = open(str(input_file))
        line = smiles_f.readline().strip()
        while line is not None and len(line) > 3:
            try:
                smiles, ids = line.split()
                mol = Chem.MolFromSmiles(smiles)
                mol.SetProp("_Name", ids)
                mol_counter += 1
                t0 = time.time()
                if mol:
                    name = mol.GetProp("_Name")
                    e = []
                    keep = []
                    n = 300
                    mol = Chem.AddHs(mol)
                    if removehydrogens == 1:
                        mol = Chem.RemoveHs(mol)

                    nrot = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                    if nrot <= 12:
                        n = 200
                        if nrot <= 7:
                            n = 50
                    print(
                        "Mol",
                        mol_counter,
                        "title =",
                        mol.GetProp("_Name"),
                        " Nrot = ",
                        nrot,
                        ", generating",
                        numtokeep,
                        "low E confs...",
                    )
                    confIds = AllChem.EmbedMultipleConfs(mol, n)
                    for confId in confIds:
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=confId)
                        AllChem.UFFOptimizeMolecule(mol, confId=confId)
                        # ff.Minimize() # --Redundant
                        e.append(ff.CalcEnergy())
                        d = sorted(zip(e, confIds))
                    for conf in d:
                        if len(keep) < numtokeep:
                            if shouldWeKeep(keep, mol, conf[1]):
                                keep.append(conf)
                    counter = 1
                    for molout in keep:
                        mol.SetProp("_Name", name)
                        writer.write(mol, confId=molout[1])
                        counter = counter + 1
                    print("took", time.time() - t0, "seconds.")
            except:
                print("Error making a molecule from:\n" + str(line))
                print(sys.exc_info())
            line = smiles_f.readline().strip()
        writer.close()
        print("\nGenerated", numtokeep, "in", time.time() - t00, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", help="Smiles file")
    args = parser.parse_args()
    rdkonf_use_files(Path(args.smiles_file))
