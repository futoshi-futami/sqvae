import copy
import yaml

class CfgNode(dict):
    def __init__(self, init_dict=None, new_allowed=False):
        super().__init__()
        object.__setattr__(self, '_new_allowed', new_allowed)
        if init_dict is not None:
            for k, v in init_dict.items():
                if isinstance(v, dict):
                    v = CfgNode(v, new_allowed=new_allowed)
                self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        if object.__getattribute__(self, '_new_allowed'):
            node = CfgNode(new_allowed=True)
            self[name] = node
            return node
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if not object.__getattribute__(self, '_new_allowed') and name not in self:
            raise AttributeError(name)
        if isinstance(value, dict) and not isinstance(value, CfgNode):
            value = CfgNode(value, new_allowed=object.__getattribute__(self, '_new_allowed'))
        self[name] = value

    # dictionary .get is inherited

    def merge_from_file(self, file_path):
        with open(file_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg:
            self.merge_from_dict(cfg)

    def merge_from_dict(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                if k not in self:
                    self[k] = CfgNode(new_allowed=True)
                self[k].merge_from_dict(v)
            else:
                setattr(self, k, v)

    def clone(self):
        return copy.deepcopy(self)

    def freeze(self):
        def _freeze(node):
            object.__setattr__(node, '_new_allowed', False)
            for val in node.values():
                if isinstance(val, CfgNode):
                    _freeze(val)
        _freeze(self)
