from mp_api.client import MPRester
with MPRester(api_key="jWg2eBfBHBR0eKd57W2Se8FvSbddci4o") as mpr:
    data = mpr.materials.search(material_ids=["mp-10070"])


with MPRester("jWg2eBfBHBR0eKd57W2Se8FvSbddci4o") as mpr:
    docs = mpr.materials.phonon.search(
        material_ids=["mp-10070"],
        fields=["material_id", "phonon_dos"]
    )

print(docs)