
# steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/


def xconfig_to_configs():
    args = get_args()
    # 备份
    backup_xconfig_file(args.xconfig_file, args.config_dir)

    existing_layers = []
    if args.existing_model is not None:
        existing_layers = xparser.get_model_component_info(args.existing_model)

    # 通过read_xconfig_file() 生成对应的xlayers
    all_layers = xparser.read_xconfig_file(args.xconfig_file, existing_layers)

    write_expanded_xconfig_files(args.config_dir, all_layers)
    
    write_config_files(args.config_dir, all_layers)
    check_model_contexts(args.config_dir, args.nnet_edits,
                         existing_model=args.existing_model)
    add_nnet_context_info(args.config_dir, args.nnet_edits,
                          existing_model=args.existing_model)





def read_xconfig_file(xconfig_filename, existing_layers=[]):
    f = open(xconfig_filename, 'r')
    all_layers = []
    while True:
        line = f.readline()
        if line == '':
            break
        # 通过解析每行, 得到 xlayer对象.
        this_layer = xconfig_line_to_object(line, existing_layers)
        if this_layer is None:
            continue  # line was blank after removing comments.
        all_layers.append(this_layer)
        existing_layers.append(this_layer)
        
    f.close()
    return all_layers


def xconfig_line_to_object(config_line, prev_layers = None):
    x  = xutils.parse_config_line(config_line)
    if x is None:
        return None
    (first_token, key_to_value) = x
    # 数组中找到对应的 xlayer对象 并根据first_token, key_to_value 构造
    return config_to_layer[first_token](first_token, key_to_value, prev_layers)







class XconfigLayerBase(object):
    """ A base-class for classes representing layers of xconfig files.
    """

    def __init__(self, first_token, key_to_value, all_layers):
        """
         first_token: first token on the xconfig line, e.g. 'affine-layer'.
         key_to_value: dictionary with parameter values
             { 'name':'affine1',
               'input':'Append(0, 1, 2, ReplaceIndex(ivector, t, 0))',
               'dim=1024' }.

        The only required and 'special' values that are dealt with directly at this level, are 'name' and 'input'. 
        The rest are put in self.config and are dealt with by the child classes' init functions.
        all_layers: An array of objects inheriting XconfigLayerBase for all  previously parsed layers.
        """
        # first_token --- get the layer_type
        self.layer_type = first_token
        if 'name' not in key_to_value:
            raise RuntimeError("Expected 'name' to be specified.")
        self.name = key_to_value['name']
        if not xutils.is_valid_line_name(self.name):
            raise RuntimeError("Invalid value: name={0}".format(
                key_to_value['name']))

        # It is possible to have two layers with a same name in 'all_layer', if
        # the layer type for one of them is 'existing'.
        # Layers of type 'existing' are corresponding to the component-node names
        # in the existing model, which we are adding layers to them.
        # 'existing' layers are not presented in any config file, and new layer
        # with the same name can exist in 'all_layers'.
        
        # e.g. It is possible to have 'output-node' with name 'output' in the
        # existing model, which is added to all_layers using layer type 'existing',
        # and 'output-node' of type 'output-layer' with the same name 'output' in
        # 'all_layers'.
        for prev_layer in all_layers:
            if (self.name == prev_layer.name and
                prev_layer.layer_type is not 'existing'):
                raise RuntimeError("Name '{0}' is used for more than one "
                                   "layer.".format(self.name))

        self.config = {}
        # the following, which should be overridden in the child class, sets
        # default config parameters in self.config.
        self.set_default_configs()

        # 设置配置值 为具体user具体用,然后解析得到具体Descriptors??
        # The following is not to be reimplemented in child classes;
        # it sets the config values to those specified by the user, and
        # parses any Descriptors.
        self.set_configs(key_to_value, all_layers)
        
        # This method, sets the derived default config values
        # i.e., config values when not specified can be derived from
        # other values. It can be overridden in the child class.
        self.set_derived_configs()
        # the following, which should be overridden in the child class, checks
        # that the config parameters that have been set are reasonable.
        self.check_configs()


    def set_configs(self, key_to_value, all_layers):
        """ Sets the config variables.
            We broke this code out of __init__ for clarity.
            the child-class constructor will deal with the configuration values
            in a more specific way.
        """

        # First check that there are no keys that don't correspond to any config
        # parameter of this layer, and if so, raise an exception with an
        # informative message saying what configs are allowed.
        for key, value in key_to_value.items():
            if key != 'name':
                if key not in self.config:
                    configs = ' '.join([('{0}->"{1}"'.format(x, y) if isinstance(y, str)
                                         else '{0}->{1}'.format(x, y))
                                        for x, y in self.config.items()])
                    raise RuntimeError("Configuration value {0}={1} was not "
                                       "expected in layer of type {2}; allowed "
                                       "configs with their defaults: {3}"
                                       "" .format(key, value, self.layer_type, configs))

        for key, value in key_to_value.items():
            if key != 'name':
                assert key in self.config  # we checked above.
                self.config[key] = xutils.convert_value_to_type(key,
                                                                type(self.config[key]),
                                                                value)
        self.descriptors = dict()
        self.descriptor_dims = dict()
        # Parse Descriptors and get their dims and their 'final' string form.
        # in self.descriptors[key]
        for key in self.get_input_descriptor_names():
            if key not in self.config:
                raise RuntimeError("{0}: object of type {1} needs to override"
                                   " get_input_descriptor_names()."
                                   "".format(sys.argv[0], str(type(self))))

            descriptor_string = self.config[key]  # input string.
            assert isinstance(descriptor_string, str)
            desc = self.convert_to_descriptor(descriptor_string, all_layers)
            desc_dim = self.get_dim_for_descriptor(desc, all_layers)
            desc_norm_str = desc.str()

            # desc_output_str contains the "final" component names, those that
            # appear in the actual config file (i.e. not names like
            # 'layer.auxiliary_output'); that's how it differs from desc_norm_str.
            # Note: it's possible that the two strings might be the same in
            # many, even most, cases-- it depends whether
            # output_name(self, auxiliary_output)
            # returns self.get_name() + '.' + auxiliary_output
            # when auxiliary_output is not None.
            # That's up to the designer of the layer type.
            desc_output_str = self.get_string_for_descriptor(desc, all_layers)
            self.descriptors[key] = {'string': desc,
                                     'normalized-string': desc_norm_str,
                                     'final-string': desc_output_str,
                                     'dim': desc_dim}

            # the following helps to check the code by parsing it again.
            desc2 = self.convert_to_descriptor(desc_norm_str, all_layers)
            desc_norm_str2 = desc2.str()
            # if the following ever fails we'll have to do some debugging.
            if desc_norm_str != desc_norm_str2:
                raise RuntimeError("Likely code error: '{0}' != '{1}'"
                                   "".format(desc_norm_str, desc_norm_str2))

    def str(self):
        """Converts 'this' to a string which could be printed to
        an xconfig file; in xconfig_to_configs.py we actually expand all the
        lines to strings and write it as xconfig.expanded as a reference
        (so users can see any defaults).
        """

        list_of_entries = ['{0} name={1}'.format(self.layer_type, self.name)]
        for key, value in sorted(self.config.items()):
            if isinstance(value, str) and re.search('=', value):
                # the value is a string that contains an '=' sign, so we need to
                # enclose it in double-quotes, otherwise we woudldn't be able to
                # parse from that output.
                if re.search('"', value):
                    print("Warning: config '{0}={1}' contains both double-quotes "
                          "and equals sign; it will not be possible to parse it "
                          "from the file.".format(key, value), file=sys.stderr)
                list_of_entries.append('{0}="{1}"'.format(key, value))
            else:
                list_of_entries.append('{0}={1}'.format(key, value))

        return ' '.join(list_of_entries)

    def __str__(self):
        return self.str()

    def normalize_descriptors(self):
        """Converts any config variables in self.config which correspond to
        Descriptors, into a 'normalized form' derived from parsing them as
        Descriptors, replacing things like [-1] with the actual layer names,
        and regenerating them as strings.  We stored this when the object was
        initialized, in self.descriptors; this function just copies them back
        to the config.
        """

        for key, desc_str_dict in self.descriptors.items():
            self.config[key] = desc_str_dict['normalized-string']

    def convert_to_descriptor(self, descriptor_string, all_layers):
        """Convenience function intended to be called from child classes,
        converts a string representing a descriptor ('descriptor_string')
        into an object of type Descriptor, and returns it. It needs 'self' and
        'all_layers' (where 'all_layers' is a list of objects of type
        XconfigLayerBase) so that it can work out a list of the names of other
        layers, and get dimensions from them.
        """

        prev_names = xutils.get_prev_names(all_layers, self)
        tokens = xutils.tokenize_descriptor(descriptor_string, prev_names)
        pos = 0
        (descriptor, pos) = xutils.parse_new_descriptor(tokens, pos, prev_names)
        # note: 'pos' should point to the 'end of string' marker
        # that terminates 'tokens'.
        if pos != len(tokens) - 1:
            raise RuntimeError("Parsing Descriptor, saw junk at end: {0}"
                               "".format(' '.join(tokens[pos:-1])))
        return descriptor

    def get_dim_for_descriptor(self, descriptor, all_layers):
        """Returns the dimension of a Descriptor object. This is a convenience
        function used in set_configs.
        """

        layer_to_dim_func = \
                lambda name: xutils.get_dim_from_layer_name(all_layers, self,
                                                            name)
        return descriptor.dim(layer_to_dim_func)

    def get_string_for_descriptor(self, descriptor, all_layers):
        """Returns the 'final' string form of a Descriptor object,
        as could be used in config files. This is a convenience function
        provided for use in child classes;
        """

        layer_to_string_func = \
                lambda name: xutils.get_string_from_layer_name(all_layers,
                                                               self, name)
        return descriptor.config_string(layer_to_string_func)

    def get_name(self):
        """Returns the name of this layer, e.g. 'affine1'.  It does not
        necessarily correspond to a component name.
        """

        return self.name

    ######  Functions that might be overridden by the child class: #####

    def set_default_configs(self):
        """Child classes should override this.
        """

        raise Exception("Child classes must override set_default_configs().")

    def set_derived_configs(self):
        """This is expected to be called after set_configs and before
        check_configs().
        """
        if 'dim' in self.config and self.config['dim'] <= 0:
            self.config['dim'] = self.descriptors['input']['dim']

    def check_configs(self):
        """child classes should override this.
        """

        pass

    def get_input_descriptor_names(self):
        """This function, which may be (but usually will not have to be)
        overridden by child classes, returns a list of names of the input
        descriptors expected by this component. Typically this would just
        return ['input'] as most layers just have one 'input'. However some
        layers might require more inputs (e.g. cell state of previous LSTM layer
        in Highway LSTMs). It is used in the function 'normalize_descriptors()'.
        This implementation will work for layer types whose only
        Descriptor-valued config is 'input'.
        If a child class adds more inputs, or does not have an input
        (e.g. the XconfigInputLayer), it should override this function's
        implementation to something like: `return ['input', 'input2']`
        """

        return ['input']

    def auxiliary_outputs(self):
        """Returns a list of all auxiliary outputs that this layer supports.
        These are either 'None' for the regular output, or a string
        (e.g. 'projection' or 'memory_cell') for any auxiliary outputs that
        the layer might provide.  Most layer types will not need to override
        this.
        """

        return [None]

    def output_name(self, auxiliary_output=None):
        """Called with auxiliary_output is None, this returns the component-node
        name of the principal output of the layer (or if you prefer, the text
        form of a descriptor that gives you such an output; such as
        Append(some_node, some_other_node)).
        The 'auxiliary_output' argument is a text value that is designed for
        extensions to layers that have additional auxiliary outputs.
        For example, to implement a highway LSTM you need the memory-cell of a
        layer, so you might allow auxiliary_output='memory_cell' for such a
        layer type, and it would return the component node or a suitable
        Descriptor: something like 'lstm3.c_t'
        """

        raise Exception("Child classes must override output_name()")

    def output_dim(self, auxiliary_output=None):
        """The dimension that this layer outputs.  The 'auxiliary_output'
        parameter is for layer types which support auxiliary outputs.
        """

        raise Exception("Child classes must override output_dim()")

    def get_full_config(self):
        """This function returns lines destined for the 'full' config format, as
        would be read by the C++ programs. Since the program
        xconfig_to_configs.py writes several config files, this function returns
        a list of pairs of the form (config_file_basename, line),
        e.g. something like
         [  ('init', 'input-node name=input dim=40'),
            ('ref', 'input-node name=input dim=40') ]
        which would be written to config_dir/init.config and config_dir/ref.config.
        """

        raise Exception("Child classes must override get_full_config()")

