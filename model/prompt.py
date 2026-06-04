CLASS_NAME_MAP = {
    # VisA — numbered variants
    'pcb1':       'printed circuit board',
    'pcb2':       'printed circuit board',
    'pcb3':       'printed circuit board',
    'pcb4':       'printed circuit board',
    'macaroni1':  'macaroni',
    'macaroni2':  'macaroni',
    # VisA — compound / abbreviated names
    'chewinggum': 'chewing gum',
    'pipe_fryum': 'pipe fryum',
    'capsules':   'capsule',
    # MVTec — names that need spacing / expansion
    'metal_nut':      'metal nut',
    'wood':           'wood',
    'zipper':         'zipper',
    'toothbrush':     'toothbrush',
    'screw':          'screw',
    'pill':           'pill',
    'leather':        'leather',
    'hazelnut':       'hazelnut',
    'grid':           'grid',
    'carpet':         'carpet',
    'bottle':         'bottle',
    'cable':          'cable',
    'capsule':        'capsule',
    'tile':           'tile',
    'transistor':     'transistor',
}

TEMPLATES = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a good photo of a {}.',
    'a close-up photo of a {}.',
    'a close-up photo of the {}.',
    'a bright photo of a {}.',
    'a dark photo of a {}.',
    'a photo of a large {}.',
    'a photo of a small {}.',
    'this is a {} in the scene.'
]

STATE_NORMAL = [
    '{}', 
    'flawless {}', 
    'perfect {}', 
    '{} without defect', 
    '{} with shadow',
    '{} on dark background',
    'empty background',
    'black background',
    'textured background',
    'detailed surface'
]

STATE_ANOMALY = [
    'damaged {}', 
    'broken {}', 
    '{} with defect', 
    'cracked {}',
    '{} with scratch',
    '{} with spot',
    '{} with irregular texture',
    'missing part in {}',
    'misplaced part in {}'
]

def get_prompts(class_name: str = 'object'):
    class_name = CLASS_NAME_MAP.get(class_name.lower(), class_name.replace('_', ' '))
    normal_sentences = []
    for state in STATE_NORMAL:
        s = state.format(class_name)
        for t in TEMPLATES:
            normal_sentences.append(t.format(s))
            
    anomaly_sentences = []
    for state in STATE_ANOMALY:
        s = state.format(class_name)
        for t in TEMPLATES:
            anomaly_sentences.append(t.format(s))
            
    return {
        'normal': normal_sentences,
        'anomaly': anomaly_sentences
    }
