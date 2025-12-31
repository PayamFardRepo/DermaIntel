from pathlib import Path

base = Path(r'C:\Users\payam\skin-classifier\backend\data\infectious_downloads\dermnet_raw\train')
infectious_folders = [
    'Cellulitis Impetigo and other Bacterial Infections',
    'Herpes HPV and other STDs Photos',
    'Nail Fungus and other Nail Disease',
    'Scabies Lyme Disease and other Infestations and Bites',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Warts Molluscum and other Viral Infections',
]
print('=== DermNet Infectious Folder Counts ===')
total = 0
for folder in infectious_folders:
    folder_path = base / folder
    if folder_path.exists():
        count = len(list(folder_path.glob('*')))
        total += count
        print(f'{folder}: {count}')
    else:
        print(f'{folder}: NOT FOUND')
print(f'\nTotal infectious images available: {total}')
