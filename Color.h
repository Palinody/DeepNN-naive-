#ifndef COLOR_H_INCLUDED
#define COLOR_H_INCLUDED

class Color
{
public:
    Color(string col);
    Color(const Color& col);
    ~Color();

protected:
    string m_color;
};

#endif // COLOR_H_INCLUDED
