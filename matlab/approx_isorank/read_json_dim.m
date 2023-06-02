function dim = read_json_dim(file)
   f   = fopen(file);
   raw = fread(f, inf);
   str = char(raw');
   val = jsondecode(str);
   dim = length(fieldnames(val));
end